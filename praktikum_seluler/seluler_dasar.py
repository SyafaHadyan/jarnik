"""
seluler_dasar.py - fungsi bersama praktikum seluler (HackRF One, hanya menerima).

Isi:
  - tabel pita seluler Indonesia (FDD/TDD, DL/UL, EARFCN)
  - pembaca rekaman IQ .cs8 (hackrf_transfer) dan CSV hackrf_sweep
  - sinyal sinkronisasi LTE: PSS (N_ID_2) dan SSS (N_ID_1), 3GPP TS 36.211 6.11
  - gaya grafik untuk slide (warna sama dengan hackrf/grafik_slide.py)
"""
import numpy as np

ORANYE, GELAP, TINTA, TINTA2, GRID = "#E8663D", "#13202B", "#13202B", "#44515C", "#E4E1D8"
BIRU, HIJAU = "#2A78D6", "#1BAF7A"
FONT = 15

# --------------------------------------------------------------------- pita seluler

# nama, mode, DL (MHz), UL (MHz), F_DL_low, N_offs_DL (EARFCN = N_offs + 10*(f - F_low))
PITA = [
    ("B28 700", "FDD", (758, 803), (703, 748), 758, 9210),
    ("B5 850", "FDD", (869, 894), (824, 849), 869, 2400),
    ("B8 900", "FDD", (925, 960), (880, 915), 925, 3450),
    ("B3 1800", "FDD", (1805, 1880), (1710, 1785), 1805, 1200),
    ("B1 2100", "FDD", (2110, 2170), (1920, 1980), 2110, 0),
    ("B40 2300", "TDD", (2300, 2400), None, 2300, 38650),
]

# lebar kanal LTE: N_RB -> (bandwidth kanal MHz, lebar terisi MHz = N_RB x 180 kHz)
LTE_BW = {6: 1.4, 15: 3, 25: 5, 50: 10, 75: 15, 100: 20}


def pita_dari_frek(f_mhz):
    """(nama, 'DL'/'UL'/'TDD', baris PITA) untuk frekuensi f, atau (None, None, None)."""
    for b in PITA:
        nama, mode, dl, ul = b[:4]
        if dl[0] <= f_mhz <= dl[1]:
            return nama, ("TDD" if mode == "TDD" else "DL"), b
        if ul and ul[0] <= f_mhz <= ul[1]:
            return nama, "UL", b
    return None, None, None


def earfcn(f_mhz):
    """EARFCN downlink untuk frekuensi tengah f (MHz), None bila di luar pita DL."""
    nama, arah, b = pita_dari_frek(f_mhz)
    if b is None or arah == "UL":
        return None
    return int(round(b[5] + 10 * (f_mhz - b[4])))


def bw_lte(lebar_mhz):
    """Tebak kelas bandwidth LTE dari lebar terisi (MHz): (N_RB, bandwidth MHz)."""
    nrb = min(LTE_BW, key=lambda n: abs(n * 0.18 - lebar_mhz))
    return nrb, LTE_BW[nrb]


# --------------------------------------------------------------------- berkas

def baca_cs8(path, fs, buang_ms=10.0, maks_s=None):
    """IQ int8 berselang (hackrf_transfer) -> complex64, DC offset dibuang.
    buang_ms: transien awal HackRF dibuang."""
    n0 = int(fs * buang_ms / 1000) * 2
    hitung = -1 if maks_s is None else int(fs * maks_s) * 2
    raw = np.fromfile(path, np.int8, count=hitung if hitung < 0 else hitung + n0)[n0:]
    klip = np.mean(np.abs(raw) >= 127)
    x = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    x -= x.mean()
    return x.astype(np.complex64), klip


def cek_klip(klip, path):
    if klip > 1e-4:
        print("PERINGATAN %s: %.2f%% sampel mentok +-127 (clipping). Turunkan -l/-g lalu rekam ulang."
              % (path, 100 * klip))


def baca_sweep(path):
    """CSV hackrf_sweep -> (frekuensi MHz, matriks dB [sapuan x bin])."""
    baris = {}
    for ln in open(path):
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 7:
            continue
        try:
            lo, lebar = float(p[2]), float(p[4])
            db = [float(x) for x in p[6:]]
        except ValueError:
            continue
        kunci = p[0] + " " + p[1]
        for i, v in enumerate(db):
            baris.setdefault(round((lo + (i + 0.5) * lebar) / 1e3), []).append((kunci, v))
    f = np.array(sorted(baris))
    n = min(len(v) for v in baris.values())
    m = np.array([[v for _, v in baris[k][:n]] for k in f]).T
    return f / 1e3, m


# --------------------------------------------------------------------- LTE PSS / SSS

PSS_ROOT = {0: 25, 1: 29, 2: 34}
OFF62 = np.r_[np.arange(-31, 0), np.arange(1, 32)]     # 62 subcarrier di sekitar DC (DC kosong)


def pss_frek(nid2):
    """Barisan Zadoff-Chu PSS (62 nilai kompleks)."""
    u = PSS_ROOT[nid2]
    n = np.arange(62)
    return np.where(n < 31, np.exp(-1j * np.pi * u * n * (n + 1) / 63),
                    np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))


def pss_waktu(nid2, nfft):
    g = np.zeros(nfft, complex)
    g[OFF62 % nfft] = pss_frek(nid2)
    return np.fft.ifft(g)


def _mseq(taps):
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(sum(x[i + t] for t in taps) % 2)
    return 1 - 2 * np.array(x)


_S, _C, _Z = _mseq([0, 2]), _mseq([0, 3]), _mseq([0, 1, 2, 4])


def sss_frek(nid1, nid2, subframe5):
    """Barisan SSS (62 nilai +-1) untuk N_ID_1, N_ID_2; subframe5=True untuk subframe 5."""
    q1 = nid1 // 30
    q = (nid1 + q1 * (q1 + 1) // 2) // 30
    m1_ = nid1 + q * (q + 1) // 2
    m0 = m1_ % 31
    m1 = (m0 + m1_ // 31 + 1) % 31
    n = np.arange(31)
    s0, s1 = _S[(n + m0) % 31], _S[(n + m1) % 31]
    c0, c1 = _C[(n + nid2) % 31], _C[(n + nid2 + 3) % 31]
    z0, z1 = _Z[(n + m0 % 8) % 31], _Z[(n + m1 % 8) % 31]
    d = np.empty(62)
    if not subframe5:
        d[0::2], d[1::2] = s0 * c0, s1 * c1 * z0
    else:
        d[0::2], d[1::2] = s1 * c0, s0 * c1 * z1
    return d


# --------------------------------------------------------------------- grafik

def koma(teks):
    """Desimal gaya Indonesia: 11.5 -> 11,5."""
    return teks.replace(".", ",")


def gaya(ax, grid_y=True):
    if grid_y:
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=FONT - 2)
    ax.xaxis.label.set_color(TINTA2)
    ax.yaxis.label.set_color(TINTA2)
    ax.xaxis.label.set_size(FONT)
    ax.yaxis.label.set_size(FONT)


def judul(ax, teks):
    ax.set_title(teks, loc="left", fontsize=FONT + 1, color=TINTA)


def spektrogram(x, nfft, hop=None):
    """|STFT|^2 dalam dB, sumbu frekuensi digeser (baris = frekuensi)."""
    hop = hop or nfft
    m = (len(x) - nfft) // hop + 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(m)[:, None]
    S = np.fft.fftshift(np.fft.fft(x[idx] * np.hanning(nfft), axis=1), axes=1)
    return 10 * np.log10(np.abs(S.T) ** 2 + 1e-6)
