#!/usr/bin/env python3
"""
cari_sel_lte.py - Percobaan 4: cell search LTE (PSS + SSS -> Physical Cell ID) dari rekaman .cs8.

Hanya memakai sinyal sinkronisasi broadcast yang dipancarkan untuk semua ponsel (TS 36.211 6.11):
  PSS (Zadoff-Chu, 62 subcarrier tengah) -> N_ID_2 (0..2), timing 5 ms
  SSS (dua m-sequence berselang)          -> N_ID_1 (0..167), awal frame 10 ms, FDD/TDD
  PCI = 3 x N_ID_1 + N_ID_2  (0..503)
Tidak ada data pengguna yang didekode.

Rekaman harus berpusat di frekuensi tengah carrier (PSS ada di 1,08 MHz tengah):
  hackrf_transfer -r data/lte1870.cs8 -f 1870000000 -s 20000000 -b 20000000 -n 20000000 \
      -l 32 -g 24 -a 0

Analisis:
  python3 cari_sel_lte.py data/lte1870.cs8 --fc 1870 --out gambar/p4_lte1870

Langkah:
  1. Turunkan ke 1,92 Msps (FFT 128, cukup untuk 6 RB tengah).
  2. Korelasikan dengan 3 PSS sambil mencoba offset frekuensi (CFO) +-50 kHz: osilator HackRF
     meleset beberapa ppm, di 1,8 GHz itu puluhan kHz, lebih besar dari spasi subcarrier 15 kHz.
  3. Lipat korelasi tiap 5 ms -> posisi PSS tiap sel.
  4. Ambil simbol SSS (FDD: 1 simbol sebelum PSS, TDD: 3 simbol sebelum), ekualisasi dengan
     kanal dari PSS, cocokkan dengan 168 x 2 kandidat (subframe 0 / 5).
  5. Baca MIB dari PBCH (subframe 0, 6 RB tengah) di beberapa frame: bandwidth sel, jumlah port
     antena, PHICH, nomor frame (SFN). MIB hanya diterima bila CRC-nya cocok, sehingga sekaligus
     membuktikan PCI dan timing frame benar. Bila SSS meragukan, ke-168 N_ID_1 dicoba satu per satu
     dan hanya PCI yang lolos CRC yang dilaporkan (+-20 detik per sel; lewati dengan --tanpa-mib).
     Dekoder PBCH ada di ../lte_cell_search/lte_common.py.
Keluaran:
  _pss.png   korelasi PSS 25 ms per N_ID_2: puncak tiap 5 ms
  _sel.csv   PCI, N_ID_1, N_ID_2, FDD/TDD, CFO, daya relatif, isi MIB
"""
import argparse
import csv
import os
import sys
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

from seluler_dasar import (FONT, GELAP, ORANYE, TINTA, TINTA2, OFF62, baca_cs8, cek_klip, earfcn, gaya,
                           judul, koma, pss_frek, pss_waktu, sss_frek)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lte_cell_search"))
try:
    import lte_common as lte
except ImportError:
    lte = None

FS = 1.92e6
NFFT = 128
SETENGAH = 9600                       # 5 ms di 1,92 Msps
JARAK_SSS = {"FDD": NFFT + 9,         # SSS = simbol 5, PSS = simbol 6 slot 0 (CP normal 9 sampel)
             "TDD": 3 * (NFFT + 9) + 1}  # SSS = simbol terakhir subframe 0, PSS = simbol 2 subframe 1
BIN62 = OFF62 % NFFT


def korelasi_pss(z, nid2):
    p = pss_waktu(nid2, NFFT)
    c = np.correlate(z, p, "valid")
    pw = np.convolve(np.abs(z) ** 2, np.ones(NFFT), "valid")
    return np.abs(c) ** 2 / (pw * np.vdot(p, p).real + 1e-12), c


def geser(z, cfo):
    return z * np.exp(-2j * np.pi * cfo / FS * np.arange(len(z)))


def cari_cfo(z, langkah=2500, rentang=50e3):
    """Coba CFO di grid, pilih yang memberi puncak PSS tertinggi (dari ketiga N_ID_2)."""
    terbaik = (0, 0.0)
    seg = z[:int(FS * 0.025)]
    for cfo in np.arange(-rentang, rentang + 1, langkah):
        zz = geser(seg, cfo)
        s = max(korelasi_pss(zz, k)[0].max() for k in range(3))
        if s > terbaik[0]:
            terbaik = (s, cfo)
    return terbaik[1]


def cfo_halus(z, posisi, nid2):
    """Beda fase antara korelasi paruh pertama dan kedua PSS -> sisa CFO (Hz)."""
    p = pss_waktu(nid2, NFFT)
    h = NFFT // 2
    acc = 0
    for i in posisi:
        a = np.vdot(p[:h], z[i:i + h])
        b = np.vdot(p[h:], z[i + h:i + NFFT])
        acc += b * np.conj(a)
    return np.angle(acc) / (2 * np.pi * h / FS)


def deteksi_sss(z, posisi, nid2, mode):
    """Akumulasi skor (N_ID_1, paritas) atas semua kemunculan PSS. Kembalikan
    (nid1, sf0_pertama, skor_terbaik/skor_kedua)."""
    kand = np.array([[sss_frek(n1, nid2, sf5) for n1 in range(168)] for sf5 in (False, True)])  # 2x168x62
    acc = np.zeros((168, 2))
    pss = pss_frek(nid2)
    for j, i in enumerate(posisi):
        s0 = i - JARAK_SSS[mode]
        if s0 < 0 or i + NFFT > len(z):
            continue
        Yp = np.fft.fft(z[i:i + NFFT])[BIN62]
        Ys = np.fft.fft(z[s0:s0 + NFFT])[BIN62]
        H = Yp * np.conj(pss)
        d = Ys * np.conj(H)
        skor = np.abs(kand @ d)                     # 2 x 168
        par = j % 2
        acc[:, 0] += skor[par]                      # hipotesis: kemunculan pertama = subframe 0
        acc[:, 1] += skor[1 - par]                  # hipotesis: kemunculan pertama = subframe 5
    rata = acc.ravel()
    urut = np.argsort(rata)[::-1]
    n1, h = np.unravel_index(urut[0], acc.shape)
    return int(n1), int(h) == 0, rata[urut[0]] / max(rata[urut[1]], 1e-12), rata[urut[0]]


def timing_pss(x3, perkiraan, nid2, cari=6):
    """Posisi PSS di 3,84 Msps, dipertajam di sekitar perkiraan dari 1,92 Msps."""
    p = lte.pss_time(nid2)
    lo = max(0, perkiraan - cari)
    c = np.abs(np.correlate(x3[lo:perkiraan + cari + lte.NFFT], p, "valid"))
    return lo + int(np.argmax(c))


SF_LAIN = {0: 5, 5: 0, 1: 6, 6: 1}      # PSS di setengah frame lainnya


def baca_mib(x3, pss3, mode, sf_pss, pci, n_frame):
    """Decode MIB di n_frame frame berturut-turut. pss3: PSS di subframe sf_pss (0/5 FDD, 1/6 TDD).
    HackRF kadang kehilangan blok sampel (USB) pada rekaman panjang, jadi posisi PSS diikuti tiap frame;
    bila PSS hilang, PSS dicari ulang dan kedua kemungkinan setengah-frame diuji lewat CRC MIB.
    Jendela FFT dimundurkan 3 sampel ke dalam CP; pergeseran fasanya diserap estimasi kanal."""
    pss = lte.pss_time(pci % 3)
    p, sf = pss3, sf_pss
    while lte.frame_start(p, mode, sf) - 3 < 0:
        p += lte.HALF_FRAME
        sf = SF_LAIN[sf]
    ref, hasil, posisi = None, [], []
    while len(hasil) < n_frame and p + lte.FRAME < len(x3):
        c = np.abs(np.correlate(x3[p - 8:p + 8 + lte.NFFT], pss, "valid"))
        k = int(np.argmax(c))
        ref = c[k] if ref is None else ref
        calon = []
        if c[k] >= 0.4 * ref:                       # PSS masih di tempatnya: pertajam saja
            p += k - 8
            calon = [(p, sf)]
        else:                                      # sampel hilang: cari ulang dalam 10 ms ke depan
            lo = max(0, p - lte.HALF_FRAME)
            c = np.abs(np.correlate(x3[lo:p + lte.FRAME + lte.NFFT], pss, "valid"))
            p = lo + int(np.argmax(c))
            calon = [(p, sf), (p, SF_LAIN[sf])]
        m = None
        posisi.append(lte.frame_start(calon[0][0], mode, calon[0][1]) - 3)
        for pc, sc in calon:
            sf0 = lte.frame_start(pc, mode, sc) - 3
            if sf0 < 0 or sf0 + lte.SUBFRAME > len(x3):
                continue
            m = lte.decode_mib(x3, sf0, pci)
            if m:
                m["t_ms"] = (sf0 + 3) / lte.FS * 1e3      # awal frame ber-SFN ini, dari awal rekaman
                p, sf = pc, sc
                posisi[-1] = sf0
                break
        hasil.append(m)
        p += lte.FRAME
    # frame yang gagal sendiri: gabungkan PBCH per kelompok 4 frame (satu periode 40 ms)
    for a0 in range(0, len(hasil), 4):
        if all(hasil[a0:a0 + 4]) or len(hasil[a0:a0 + 4]) < 2:
            continue
        sf0s = [s0 for s0 in posisi[a0:a0 + 4] if 0 <= s0 and s0 + lte.SUBFRAME <= len(x3)]
        if len(sf0s) < len(posisi[a0:a0 + 4]):
            continue
        for j, m in enumerate(lte.decode_mib_gabung(x3, sf0s, pci)):
            if m and hasil[a0 + j] is None:
                m["t_ms"] = (sf0s[j] + 3) / lte.FS * 1e3
                hasil[a0 + j] = m
    # hasil gabungan hanya diterima bila fase SFN-nya (SFN x 10 ms - waktu) sama dengan frame terdekat
    # yang lolos sendiri; ini menolak pembagian periode 40 ms yang meleset satu frame
    sendiri = [m for m in hasil if m and not m.get("gabung")]
    fase = lambda m: (m["sfn"] * 10 - m["t_ms"]) % 10240
    for i, m in enumerate(hasil):
        if m and m.get("gabung"):
            if not sendiri:
                hasil[i] = None
                continue
            dekat = min(sendiri, key=lambda r: abs(r["t_ms"] - m["t_ms"]))
            d = abs(fase(m) - fase(dekat)) % 10240
            if min(d, 10240 - d) > 2:
                hasil[i] = None
    return hasil


def paksa_pci(x3, pss3_list, nid2):
    """SSS meragukan: coba ke-168 N_ID_1, FDD/TDD, dan kedua posisi PSS dalam frame.
    Kembalikan (pci, mode, pss3, sf_pss) pertama yang MIB-nya lolos CRC di 2 frame, atau None."""
    for mode, sfs in (("FDD", (0, 5)), ("TDD", (1, 6))):
        for pss3 in pss3_list:
            for sf in sfs:
                for nid1 in range(168):
                    pci = 3 * nid1 + nid2
                    m = baca_mib(x3, pss3, mode, sf, pci, 1)
                    if m and m[0] and all(baca_mib(x3, pss3 + lte.FRAME, mode, sf, pci, 1)):
                        return pci, mode, pss3, sf
    return None


def ringkas_mib(hasil):
    ok = [m for m in hasil if m]
    if not ok:
        return "MIB gagal (%d frame)" % len(hasil) if hasil else ""
    m = ok[0]
    gabung = sum(1 for x in ok if x.get("gabung"))
    return koma("%g MHz, %d port, SFN %d-%d (%d/%d%s)" % (m["mhz"], m["ports"], ok[0]["sfn"], ok[-1]["sfn"],
                                                         len(ok), len(hasil),
                                                         ", %d lewat gabungan 40 ms" % gabung if gabung else ""))


def grafik_pss(z, sel, path, ms=25):
    n = int(FS * ms / 1000)
    fig, ax = plt.subplots(3, 1, figsize=(16, 7.6), sharex=True, sharey=True)
    for k in range(3):
        c, _ = korelasi_pss(z[:n + NFFT], k)
        t = np.arange(len(c)) / FS * 1e3
        a = ax[k]
        a.plot(t, c, color=GELAP, lw=0.7)
        teks = ["N_ID_2 = %d" % k]
        for s in sel:
            if s["nid2"] == k:
                mib = [m for m in s.get("mib", []) if m]
                tambahan = koma(" · %g MHz · %d port" % (mib[0]["mhz"], mib[0]["ports"])) if mib else ""
                teks.append("PCI %d (N_ID_1 %d)%s" % (s["pci"], s["nid1"], tambahan) if s["yakin"]
                            else "PSS terdeteksi, SSS belum jelas")
                for m in range(int(ms / 5) + 1):
                    tt = (s["pos"] + m * SETENGAH) / FS * 1e3
                    if tt < ms:
                        a.plot(tt, c[int(round(tt * FS / 1e3))] + 0.08, "v", color=ORANYE, ms=10)
        a.text(0.005, 0.92, "   ".join(teks), transform=a.transAxes, va="top", fontsize=FONT,
               color=TINTA, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        a.set_ylim(0, 1)
        gaya(a)
    ax[1].set_ylabel("korelasi PSS ternormalisasi")
    ax[-1].set_xlabel("waktu (ms) - segitiga oranye: PSS terdeteksi, berulang tiap 5 ms")
    fig.tight_layout(h_pad=0.8)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def cari_sel(x, fs, cfo_maks_khz=50.0, ambang=0.2, frame_mib=8, tanpa_mib=False, paksa=True, cetak=print,
             deteksi_ms=300):
    """Cell search pada rekaman x (complex, fs, sudah berpusat di tengah carrier).
    paksa: bila SSS meragukan, coba ke-168 N_ID_1 lewat CRC MIB (lambat).
    PSS/SSS dicari di deteksi_ms pertama saja (sebelum kemungkinan sampel hilang); MIB memakai seluruh x.
    Kembalikan (daftar sel, z 1,92 Msps setelah CFO kasar dibuang, CFO kasar Hz)."""
    r = Fraction(FS / fs).limit_denominator(1000)
    z = resample_poly(x, r.numerator, r.denominator).astype(np.complex64)
    cfo0 = cari_cfo(z, rentang=cfo_maks_khz * 1e3)
    z = geser(z, cfo0)

    sel = []
    z_semua, z = z, z[:int(FS * deteksi_ms / 1000)]
    n_lipat = len(z) // SETENGAH - 1
    for k in range(3):
        c, _ = korelasi_pss(z, k)
        lipat = c[:n_lipat * SETENGAH].reshape(n_lipat, SETENGAH).mean(axis=0)
        kandidat = []
        for i in np.argsort(lipat)[::-1]:
            if lipat[i] < ambang * 0.5 or len(kandidat) == 2:
                break
            if all(abs(i - j) > 20 for j in kandidat):
                kandidat.append(int(i))
        for pos in kandidat:
            posisi = [pos + m * SETENGAH for m in range(n_lipat) if pos + m * SETENGAH + NFFT < len(z)]
            puncak = np.mean([c[i] for i in posisi])
            if puncak < ambang:
                continue
            df = cfo_halus(z, posisi, k)
            hasil = {}
            for mode in ("FDD", "TDD"):
                hasil[mode] = deteksi_sss(geser(z, df), posisi, k, mode)
            mode = max(hasil, key=lambda m: hasil[m][3])
            n1, sf0_pertama, rasio, _ = hasil[mode]
            awal = pos if sf0_pertama else pos + SETENGAH
            daya = np.mean([np.mean(np.abs(z[i:i + NFFT]) ** 2) for i in posisi])
            sel.append(dict(pci=3 * n1 + k, nid1=n1, nid2=k, mode=mode, skor_pss=puncak, sss_rasio=rasio,
                            cfo=cfo0 + df, pos=pos, awal=awal, awal_frame_ms=awal / FS * 1e3, daya=daya,
                            yakin=rasio >= 1.5, sumber_pci="SSS", mib=[]))
    z = z_semua
    if not sel or tanpa_mib:
        return sel, z, cfo0
    if lte is None:
        cetak("lte_common.py tidak ditemukan di ../lte_cell_search: MIB dilewati.")
        return sel, z, cfo0

    r3 = Fraction(lte.FS / fs).limit_denominator(1000)
    x3_asli = resample_poly(x, r3.numerator, r3.denominator)
    for s in sel:
        x3 = x3_asli * np.exp(-2j * np.pi * s["cfo"] / lte.FS * np.arange(len(x3_asli)))
        if s["yakin"]:
            pss3 = timing_pss(x3, 2 * s["awal"], s["nid2"])
            s["mib"] = baca_mib(x3, pss3, s["mode"], 0 if s["mode"] == "FDD" else 1, s["pci"], frame_mib)
            if any(s["mib"]) or not paksa:
                continue
            cetak("PCI %d dari SSS tidak lolos CRC MIB; mencoba 168 N_ID_1 ..." % s["pci"])
            s["yakin"] = False
        elif not paksa:
            continue
        else:
            cetak("N_ID_2 %d: SSS meragukan, mencoba 168 N_ID_1 dengan CRC MIB ..." % s["nid2"])
        ketemu = paksa_pci(x3, [timing_pss(x3, 2 * s["pos"], s["nid2"])], s["nid2"])
        if ketemu:
            pci, mode, p3, sf = ketemu
            s.update(pci=pci, nid1=pci // 3, mode=mode, yakin=True, sumber_pci="CRC MIB")
            s["mib"] = baca_mib(x3, p3, mode, sf, pci, frame_mib)
        else:
            cetak("  tidak ada PCI yang lolos CRC: kemungkinan bukan sel LTE (mis. 5G NR) atau terlalu lemah")
    # PCI yang sama dari dua puncak PSS (pantulan/multipath): simpan yang MIB-nya paling banyak lolos
    terbaik = {}
    for s in sel:
        kunci = s["pci"] if s["yakin"] else ("?", s["nid2"], s["pos"])
        if kunci not in terbaik or sum(1 for m in s["mib"] if m) > sum(1 for m in terbaik[kunci]["mib"] if m):
            terbaik[kunci] = s
    sel = [s for s in sel if terbaik.get(s["pci"] if s["yakin"] else ("?", s["nid2"], s["pos"])) is s]
    return sel, z, cfo0


def main():
    p = argparse.ArgumentParser(description="Cell search LTE (PSS/SSS) dari rekaman .cs8")
    p.add_argument("rekaman")
    p.add_argument("--fs", type=float, default=20e6)
    p.add_argument("--fc", type=float, default=None, help="frekuensi tengah (MHz): untuk ppm dan EARFCN")
    p.add_argument("--geser-khz", type=float, default=0.0, help="pusat carrier relatif pusat rekaman")
    p.add_argument("--ms", type=float, default=400, help="panjang rekaman yang dianalisis (ms)")
    p.add_argument("--ambang", type=float, default=0.2, help="skor PSS minimum (0..1)")
    p.add_argument("--cfo-maks-khz", type=float, default=50.0,
                   help="rentang pencarian CFO; persempit (mis. 10) bila ppm HackRF sudah diketahui")
    p.add_argument("--ppm", type=float, default=None,
                   help="kesalahan osilator HackRF (dari survei atau carrier lain): CFO dibuang dulu dan pencarian "
                        "dipersempit ke +-6 kHz, menghindari kunci palsu Zadoff-Chu")
    p.add_argument("--frame-mib", type=int, default=8, help="jumlah frame 10 ms yang MIB-nya dibaca")
    p.add_argument("--tanpa-mib", action="store_true", help="lewati pembacaan MIB (lebih cepat)")
    p.add_argument("--out", default="gambar/p4")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    x, klip = baca_cs8(a.rekaman, a.fs, maks_s=a.ms / 1000)
    cek_klip(klip, a.rekaman)
    if a.geser_khz:
        x = x * np.exp(-2j * np.pi * a.geser_khz * 1e3 / a.fs * np.arange(len(x)))
    cfo_ppm, cfo_maks = 0.0, a.cfo_maks_khz
    if a.ppm is not None and a.fc:
        cfo_ppm, cfo_maks = a.ppm * a.fc, min(a.cfo_maks_khz, 6.0)
        x = x * np.exp(-2j * np.pi * cfo_ppm / a.fs * np.arange(len(x)))
    sel, z, cfo0 = cari_sel(x, a.fs, cfo_maks, a.ambang, a.frame_mib, a.tanpa_mib)
    cfo0 += cfo_ppm
    for s in sel:
        s["cfo"] += cfo_ppm
    print("%s: %s ms, CFO kasar %+.1f kHz" % (a.rekaman, koma("%.0f" % (len(z) / FS * 1e3)), cfo0 / 1e3))

    if not sel:
        print("Tidak ada PSS di atas ambang. Pastikan rekaman berpusat di tengah carrier LTE.")
        return
    ref = max(s["daya"] for s in sel)
    print("\n%-5s %-7s %-7s %-5s %-9s %-10s %-11s %-6s %s" % ("PCI", "N_ID_1", "N_ID_2", "mode", "skor PSS",
                                                                "yakin SSS", "CFO", "ppm", "MIB"))
    with open(a.out + "_sel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pci", "nid1", "nid2", "mode", "skor_pss", "sss_rasio", "cfo_hz", "ppm", "awal_frame_ms",
                    "daya_rel_db", "earfcn", "sumber_pci", "mib_lolos", "mib_frame", "nrb", "bw_mhz", "port",
                    "phich", "sfn_awal"])
        for s in sorted(sel, key=lambda s: -s["skor_pss"]):
            ppm = s["cfo"] / (a.fc * 1e6) * 1e6 if a.fc else float("nan")
            yakin = s["yakin"]
            ok = [m for m in s["mib"] if m]
            m0 = ok[0] if ok else {}
            w.writerow([s["pci"] if yakin else "", s["nid1"] if yakin else "", s["nid2"],
                        s["mode"] if yakin else "", "%.3f" % s["skor_pss"], "%.2f" % s["sss_rasio"],
                        "%.0f" % s["cfo"], "%.2f" % ppm, "%.3f" % s["awal_frame_ms"],
                        "%.1f" % (10 * np.log10(s["daya"] / ref)), earfcn(a.fc) if a.fc else "",
                        s["sumber_pci"] if yakin else "", len(ok), len(s["mib"]), m0.get("nrb", ""),
                        m0.get("mhz", ""), m0.get("ports", ""),
                        ("%s/%s" % (m0["phich_duration"], m0["phich_resource"])) if m0 else "", m0.get("sfn", "")])
            print("%-5s %-7s %-7d %-5s %-9s %-10s %-11s %-6s %s" % (
                s["pci"] if yakin else "?", s["nid1"] if yakin else "?", s["nid2"], s["mode"] if yakin else "?",
                koma("%.2f" % s["skor_pss"]), koma("%.1fx" % s["sss_rasio"]) + ("*" if s["sumber_pci"] != "SSS" else ""),
                koma("%+.2f kHz" % (s["cfo"] / 1e3)), koma("%+.1f" % ppm) if a.fc else "-", ringkas_mib(s["mib"])))
    if a.fc:
        print("\nEARFCN %s. Bandingkan PCI & EARFCN dengan aplikasi info sel di ponsel." % earfcn(a.fc))
    print("yakin SSS = skor kandidat terbaik / kedua terbaik; di bawah 1,5x PCI ditulis '?' "
          "(PSS terlihat, SSS tidak cukup jelas). * = PCI ditemukan lewat CRC MIB, bukan SSS.")
    print("MIB = isi PBCH yang lolos CRC: bandwidth sel, jumlah port antena, SFN (lolos/dicoba frame).")
    grafik_pss(z, sel, a.out + "_pss.png")
    print("Selesai: %s_*" % a.out)


if __name__ == "__main__":
    main()
