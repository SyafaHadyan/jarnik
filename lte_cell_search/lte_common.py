"""
lte_common.py - LTE downlink building blocks for cell search (3GPP TS 36.211 / 36.212).

Only the public broadcast signals every phone uses to find a cell:
  PSS  primary synchronisation signal   -> N_ID_2 (0..2), 5 ms timing
  SSS  secondary synchronisation signal -> N_ID_1 (0..167), 10 ms frame timing, FDD/TDD
  CRS  cell-specific reference signals  -> channel estimate
  PBCH physical broadcast channel       -> MIB: bandwidth, PHICH config, SFN, antenna ports

Numerology used here: 3.84 Msps, FFT 256, 15 kHz subcarriers, normal cyclic prefix.
"""
import numpy as np

FS = 3.84e6
NFFT = 256
CP0 = 160 * NFFT // 2048          # 20 samples, first symbol of a slot
CP = 144 * NFFT // 2048           # 18 samples, other symbols
SLOT = 7 * NFFT + CP0 + 6 * CP    # 1920 samples = 0.5 ms
SUBFRAME = 2 * SLOT               # 3840 = 1 ms
FRAME = 10 * SUBFRAME             # 38400 = 10 ms
HALF_FRAME = FRAME // 2           # PSS repeats every 5 ms
NRB_MAX = 110

PSS_ROOT = {0: 25, 1: 29, 2: 34}
MIB_NRB = [6, 15, 25, 50, 75, 100]
NRB_MHZ = {6: 1.4, 15: 3, 25: 5, 50: 10, 75: 15, 100: 20}
PHICH_RES = ["1/6", "1/2", "1", "2"]

# useful-part start of each OFDM symbol inside a slot (after its cyclic prefix)
SYM_START = []
_pos = 0
for _l in range(7):
    _c = CP0 if _l == 0 else CP
    SYM_START.append(_pos + _c)
    _pos += _c + NFFT

# distance from the SSS symbol to the PSS symbol (useful starts), per duplex mode
#   FDD: SSS = slot 0 symbol 5, PSS = slot 0 symbol 6 (subframes 0 and 5)
#   TDD: SSS = subframe 0 slot 1 symbol 6, PSS = subframe 1 slot 0 symbol 2 (DwPTS)
SSS_TO_PSS = {"FDD": SYM_START[6] - SYM_START[5],
              "TDD": SYM_START[2] - (SYM_START[6] - SLOT)}


def sym_pos(l_sub):
    """Useful-part start of OFDM symbol l (0..13) relative to the subframe start."""
    return (l_sub // 7) * SLOT + SYM_START[l_sub % 7]


# --------------------------------------------------------------------- subcarrier maps

OFF62 = np.r_[np.arange(-31, 0), np.arange(1, 32)]     # PSS/SSS: 62 subcarriers, DC unused
OFF72 = np.r_[np.arange(-36, 0), np.arange(1, 37)]     # centre 6 resource blocks
BIN62 = OFF62 % NFFT
BIN72 = OFF72 % NFFT


# --------------------------------------------------------------------- PSS

def pss_freq(nid2):
    u = PSS_ROOT[nid2]
    n = np.arange(62)
    return np.where(n < 31, np.exp(-1j * np.pi * u * n * (n + 1) / 63),
                    np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))


def pss_time(nid2):
    g = np.zeros(NFFT, complex)
    g[BIN62] = pss_freq(nid2)
    return np.fft.ifft(g) * np.sqrt(NFFT)


# --------------------------------------------------------------------- SSS

def _mseq(taps):
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(sum(x[i + t] for t in taps) % 2)
    return 1 - 2 * np.array(x)


_S = _mseq([2, 0])                  # x(i+5) = x(i+2) + x(i)
_C = _mseq([3, 0])                  # x(i+5) = x(i+3) + x(i)
_Z = _mseq([4, 2, 1, 0])            # x(i+5) = x(i+4) + x(i+2) + x(i+1) + x(i)


def sss_freq(nid1, nid2, subframe):
    """62 SSS chips (+-1) for subframe 0 or 5 (TS 36.211 6.11.2.1)."""
    q1 = nid1 // 30
    q = (nid1 + q1 * (q1 + 1) // 2) // 30
    m_ = nid1 + q * (q + 1) // 2
    m0 = m_ % 31
    m1 = (m0 + m_ // 31 + 1) % 31
    n = np.arange(31)
    s0, s1 = _S[(n + m0) % 31], _S[(n + m1) % 31]
    c0, c1 = _C[(n + nid2) % 31], _C[(n + nid2 + 3) % 31]
    z0, z1 = _Z[(n + m0 % 8) % 31], _Z[(n + m1 % 8) % 31]
    d = np.empty(62)
    if subframe == 0:
        d[0::2], d[1::2] = s0 * c0, s1 * c1 * z0
    else:
        d[0::2], d[1::2] = s1 * c0, s0 * c1 * z1
    return d


def sss_table(nid2):
    """All SSS candidates for one N_ID_2: [2, 168, 62], index 0 = subframe 0, 1 = subframe 5."""
    return np.array([[sss_freq(n1, nid2, sf) for n1 in range(168)] for sf in (0, 5)])


def detect_sss(y_sss, y_pss, nid2):
    """Match one received SSS symbol against all 168 x 2 candidates.
    y_sss, y_pss: 62 centre subcarriers (BIN62) of the SSS and PSS symbols; the PSS gives the
    channel. Returns (nid1, subframe 0/5, best / second-best score)."""
    h = y_pss * np.conj(pss_freq(nid2))
    score = np.abs(sss_table(nid2) @ (y_sss * np.conj(h)))
    order = np.argsort(score.ravel())[::-1]
    sf, nid1 = np.unravel_index(order[0], score.shape)
    return int(nid1), 5 * int(sf), score.ravel()[order[0]] / max(score.ravel()[order[1]], 1e-12)


def frame_start(pss_pos, duplex, subframe):
    """Start (CP included) of subframe 0 for a PSS whose useful part starts at pss_pos.
    subframe: where that PSS sits, 0/5 (FDD) or 1/6 (TDD), as found from the SSS."""
    l_pss = 6 if duplex == "FDD" else 2
    return pss_pos - sym_pos(l_pss) - subframe * SUBFRAME


# --------------------------------------------------------------------- Gold sequence, CRS

def gold(c_init, n):
    """Pseudo-random sequence c(0..n-1) of TS 36.211 7.2 (Nc = 1600), as 0/1."""
    nc = 1600
    x1 = np.zeros(nc + n + 31, np.uint8)
    x2 = np.zeros(nc + n + 31, np.uint8)
    x1[0] = 1
    x2[:31] = (c_init >> np.arange(31)) & 1
    for i in range(nc + n):
        x1[i + 31] = x1[i + 3] ^ x1[i]
        x2[i + 31] = x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]
    return x1[nc:nc + n] ^ x2[nc:nc + n]


def crs(ns, l, pci, port, nrb=6):
    """Cell-specific reference signal of one antenna port (TS 36.211 6.10.1), normal CP.
    Returns (k, values) for the centre nrb RBs, k = subcarrier index -6*nrb..6*nrb-1 counted
    without DC (k >= 0 is above DC); empty arrays if the port has no CRS in slot ns, symbol l.
    The sequence is centred on DC, so this does not depend on the cell bandwidth."""
    if port < 2 and l in (0, 4):
        v = 3 * ((port + (l == 4)) % 2)
    elif port >= 2 and l == 1:
        v = 3 * (ns % 2) + 3 * (port == 3)
    else:
        return np.array([], int), np.array([], complex)
    c_init = (2 ** 10 * (7 * (ns + 1) + l + 1) * (2 * pci + 1) + 2 * pci + 1) % 2 ** 31
    c = gold(c_init, 4 * NRB_MAX)
    r = ((1 - 2.0 * c[0::2]) + 1j * (1 - 2.0 * c[1::2])) / np.sqrt(2)
    k = 6 * (np.arange(2 * NRB_MAX) - NRB_MAX) + (v + pci % 6) % 6
    keep = (k >= -6 * nrb) & (k < 6 * nrb)
    return k[keep], r[keep]


# --------------------------------------------------------------------- OFDM, channel

def ofdm_subframe(x, start):
    """FFT of the 14 symbols of the subframe whose first CP starts at `start`: [14, NFFT]."""
    return np.array([np.fft.fft(x[start + sym_pos(l):start + sym_pos(l) + NFFT])
                     for l in range(14)]) / np.sqrt(NFFT)


def centre72(F):
    """Centre 6 RBs of an FFT grid [.., NFFT] -> [.., 72], index = k + 36."""
    return F[..., BIN72]


def channel_estimate(Y, pci, ns0=0):
    """Least-squares CRS estimates for ports 0..3 on a centre-72 subframe grid Y [14, 72]
    (slots ns0, ns0 + 1): linear in frequency between pilots, linear in time between CRS
    symbols (follows residual CFO). Returns H [4, 14, 72]."""
    k_all = np.arange(-36, 36)
    H = np.zeros((4, 14, 72), complex)
    for p in range(4):
        t, est = [], []
        for l_sub in range(14):
            k, r = crs(ns0 + l_sub // 7, l_sub % 7, pci, p)
            if len(k):
                h = Y[l_sub, k + 36] / r
                est.append(np.interp(k_all, k, h.real) + 1j * np.interp(k_all, k, h.imag))
                t.append(l_sub)
        est = np.array(est)
        for i in range(72):
            H[p, :, i] = (np.interp(np.arange(14), t, est[:, i].real)
                          + 1j * np.interp(np.arange(14), t, est[:, i].imag))
    return H


# --------------------------------------------------------------------- PBCH -> MIB

def pbch_re(pci):
    """(symbol, centre-72 index) of the 240 PBCH REs of subframe 0 in mapping order (TS 36.211
    6.6.4): slot 1 symbols 0..3, k first. REs of CRS ports 0..3 are skipped even if the cell
    has fewer ports."""
    li, ki = [], []
    for l_sub in (7, 8, 9, 10):
        for k in range(72):
            if l_sub in (7, 8) and k % 3 == pci % 3:
                continue
            li.append(l_sub)
            ki.append(k)
    return np.array(li), np.array(ki)


def pbch_combine(Y, H, pci, nports):
    """Undo transmit diversity on the 240 PBCH REs (TS 36.211 6.3.4.3).
    1 port: matched filter. 2 ports: SFBC (Alamouti in frequency). 4 ports: SFBC + FSTD,
    pairs alternate between ports (0, 2) and (1, 3). Output is reliability-weighted, not
    normalised: fine for the soft decoder below."""
    li, ki = pbch_re(pci)
    y, h = Y[li, ki], H[:, li, ki]
    if nports == 1:
        return y * np.conj(h[0])
    out = np.empty(240, complex)
    for i in range(0, 240, 2):
        a, b = (0, 1) if nports == 2 else ((0, 2) if (i // 2) % 2 == 0 else (1, 3))
        h0, h1 = (h[a, i] + h[a, i + 1]) / 2, (h[b, i] + h[b, i + 1]) / 2
        r0, r1 = y[i], y[i + 1]
        out[i] = np.conj(h0) * r0 + h1 * np.conj(r1)
        out[i + 1] = np.conj(h0) * r1 - h1 * np.conj(r0)
    return out


def qpsk_llr(z):
    """Soft bits of QPSK symbols, positive = bit 0 (TS 36.211 7.1.2)."""
    out = np.empty(2 * len(z))
    out[0::2], out[1::2] = z.real, z.imag
    return out


# rate matching of the tail-biting convolutional code (TS 36.212 5.1.4.2), K = 40 (24 MIB + 16 CRC)
_PERM = [1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
         0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30]


def _rm_index(K=40):
    """Circular-buffer position j (0..3K-1, NULLs removed) -> index into the coded bits [3, K]."""
    R = -(-K // 32)
    y = np.r_[-np.ones(32 * R - K, int), np.arange(K)]
    v = y.reshape(R, 32)[:, _PERM].T.ravel()
    return np.concatenate([i * K + v[v >= 0] for i in range(3)])


RM_IDX = _rm_index()
_G = (0o133, 0o171, 0o165)                  # generators, MSB = current input bit


def _conv_out():
    """Expected +-1 outputs [64 states, 2 inputs, 3]; state = 6 previous bits, newest = MSB."""
    out = np.zeros((64, 2, 3))
    for s in range(64):
        for b in (0, 1):
            reg = (b << 6) | s
            for i, g in enumerate(_G):
                out[s, b, i] = 1 - 2 * (bin(reg & g).count("1") & 1)
    return out


_OUT = _conv_out()


def viterbi_tb(soft):
    """Tail-biting Viterbi for the rate-1/3, K=7 code. soft [K, 3], positive = bit 0.
    Wrap-around decoding: run over three copies, keep the middle one."""
    K = len(soft)
    L = np.vstack([soft, soft, soft])
    ns = np.arange(64)
    b = ns >> 5
    p0 = (ns & 31) << 1
    p1 = p0 | 1
    metric = np.zeros(64)
    choice = np.zeros((3 * K, 64), np.uint8)
    for t in range(3 * K):
        bm = _OUT @ L[t]                         # [64, 2]
        m0, m1 = metric[p0] + bm[p0, b], metric[p1] + bm[p1, b]
        choice[t] = m1 > m0
        metric = np.maximum(m0, m1)
        metric -= metric.max()
    s = int(np.argmax(metric))
    bits = np.empty(3 * K, np.uint8)
    for t in range(3 * K - 1, -1, -1):
        bits[t] = s >> 5
        s = ((s & 31) << 1) | int(choice[t, s])
    return bits[K:2 * K]


def crc16(bits):
    """CRC of TS 36.212 5.1.1, g(D) = D^16 + D^12 + D^5 + 1, as a 16-bit integer."""
    reg = 0
    for b in bits:
        fb = ((reg >> 15) & 1) ^ int(b)
        reg = ((reg << 1) & 0xFFFF) ^ (0x1021 if fb else 0)
    return reg


CRC_MASK = {1: 0x0000, 2: 0xFFFF, 4: 0x5555}   # PBCH CRC mask = number of antenna ports


def parse_mib(a, frame_in_period):
    """24 MIB bits (TS 36.331) -> dict. The PBCH carries SFN / 4; the frame's place in the
    40 ms PBCH period gives the 2 low bits."""
    val = lambda bits: int("".join(str(int(v)) for v in bits), 2)
    nrb = MIB_NRB[val(a[0:3])] if val(a[0:3]) < len(MIB_NRB) else None
    return {"nrb": nrb, "mhz": NRB_MHZ.get(nrb), "phich_duration": "extended" if a[3] else "normal",
            "phich_resource": PHICH_RES[val(a[4:6])], "sfn": (val(a[6:14]) << 2) + frame_in_period}


def pbch_llr(x, sf0, pci):
    """Soft bits PBCH (480, belum di-descramble) subframe 0 di sf0, untuk 1/2/4 port: {port: llr}."""
    Y = centre72(ofdm_subframe(x, sf0))
    H = channel_estimate(Y, pci)
    hasil = {}
    for nports in (1, 2, 4):
        llr = qpsk_llr(pbch_combine(Y, H, pci, nports))
        hasil[nports] = llr / (np.mean(np.abs(llr)) + 1e-12)
    return hasil


def _decode_crc(w, nports):
    """Rate recovery + Viterbi + CRC dengan mask port. w: 120 soft bits. Kembalikan 40 bit atau None."""
    d = np.zeros(120)
    d[RM_IDX] = w
    bits = viterbi_tb(d.reshape(3, 40).T)
    if crc16(bits[:24]) ^ int("".join(map(str, bits[24:])), 2) == CRC_MASK[nports]:
        return bits
    return None


def decode_mib(x, sf0, pci):
    """Decode the MIB from the subframe 0 whose first CP starts at sample sf0.
    x must be at FS with CFO removed. Tries 1/2/4 antenna ports and the 4 places of this frame in
    the 40 ms PBCH period; only a CRC match with the antenna-port mask is accepted.
    Returns dict (with 'ports') or None."""
    llr = pbch_llr(x, sf0, pci)
    c = 1 - 2.0 * gold(pci, 1920)
    for nports in (1, 2, 4):
        for i in range(4):
            bits = _decode_crc((llr[nports] * c[480 * i:480 * (i + 1)]).reshape(4, 120).sum(axis=0), nports)
            if bits is not None:
                mib = parse_mib(bits[:24], i)
                mib["ports"] = nports
                return mib
    return None


def decode_mib_gabung(x, sf0_list, pci):
    """Gabungkan PBCH beberapa frame berurutan (subframe 0 di sf0_list, selang 10 ms).
    Dalam satu periode 40 ms isi MIB sama dan hanya segmen scrambling-nya yang berbeda, jadi soft bits
    frame-frame itu bisa dijumlahkan (SNR naik sampai 6 dB). Posisi frame pertama dalam periode tidak
    diketahui: keempat kemungkinan dicoba, hanya CRC yang cocok diterima.
    Kembalikan daftar (satu per frame) berisi dict MIB (dengan 'gabung' = jumlah frame) atau None."""
    llr = [pbch_llr(x, s, pci) for s in sf0_list]
    c = 1 - 2.0 * gold(pci, 1920)
    hasil = [None] * len(sf0_list)
    for nports in (1, 2, 4):
        for i0 in range(4):
            blok = {}
            for t in range(len(sf0_list)):
                seg = (i0 + t) % 4
                w = (llr[t][nports] * c[480 * seg:480 * (seg + 1)]).reshape(4, 120).sum(axis=0)
                blok.setdefault((i0 + t) // 4, []).append((t, seg, w))
            for anggota in blok.values():
                if len(anggota) < 2:
                    continue
                bits = _decode_crc(sum(w for _, _, w in anggota), nports)
                if bits is None:
                    continue
                for t, seg, _ in anggota:
                    mib = parse_mib(bits[:24], seg)
                    mib.update(ports=nports, gabung=len(anggota))
                    hasil[t] = mib
        if any(hasil):
            return hasil
    return hasil
