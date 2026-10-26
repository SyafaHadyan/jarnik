#!/usr/bin/env python3
"""
gsm_fcch.py - Percobaan 6: bentuk sinyal GSM (2G, TDMA) dari rekaman IQ HackRF (.cs8).

GSM membagi pita menjadi carrier 200 kHz (ARFCN); tiap carrier dibagi waktu menjadi frame TDMA
4,615 ms = 8 timeslot x 577 us. Carrier BCCH tiap sel memancarkan FCCH: burst berupa nada murni
+67,7 kHz di atas frekuensi carrier, di timeslot 0 pada frame 0, 10, 20, 30, 40 dari multiframe
51 frame. Selang FCCH karena itu 10-10-10-10-11 frame = 46,15 / 50,77 ms.
Tidak ada data pengguna yang didekode.

Rekam (sekitar carrier GSM yang terlihat sempit di Percobaan 1; 5 detik = +-21 multiframe):
  hackrf_transfer -r data/gsm951.cs8 -f 951000000 -s 4000000 -b 3500000 -n 20000000 -l 16 -g 16 -a 0

Tiap carrier diturunkan ke 400 ksps sebelum dianalisis, jadi rekaman beberapa detik tetap ringan.

Analisis:
  python3 gsm_fcch.py data/gsm951.cs8 --fc 951 --out gambar/p6_gsm

Keluaran:
  _gsm.png  (atas) spektrum: carrier 200 kHz + ARFCN, BCCH ditandai
            (tengah) kemunculan FCCH di detik pertama + selangnya
            (bawah) satu frame TDMA (8 timeslot) di carrier BCCH
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, firwin, lfilter, resample_poly, welch

from seluler_dasar import BIRU, FONT, GELAP, ORANYE, TINTA, TINTA2, baca_cs8, cek_klip, gaya, judul, koma

F_FCCH = 1625e3 / 24          # 67,708 kHz
FRAME_MS = 120 / 26           # 4,615 ms
TS_US = FRAME_MS * 1000 / 8   # 576,9 us
DESIMASI = 10                 # 4 Msps -> 400 ksps per carrier


def arfcn(f_mhz):
    """ARFCN downlink GSM 900 / E-GSM / DCS 1800, None bila di luar raster."""
    if 935.2 <= f_mhz <= 959.8:
        return int(round((f_mhz - 935) / 0.2))
    if 925.2 <= f_mhz < 935.0:
        return int(round((f_mhz - 935) / 0.2)) + 1024
    if 1805.2 <= f_mhz <= 1879.8:
        return int(round((f_mhz - 1805.2) / 0.2)) + 512
    return None


def cari_carrier(f, p_db, fc, ambang_db=10):
    k = max(1, int(60e3 / (f[1] - f[0])))      # haluskan 60 kHz: cari punuk 200 kHz, bukan garis nada
    halus = 10 * np.log10(np.convolve(10 ** (p_db / 10), np.ones(k) / k, "same"))
    nf = np.percentile(halus, 30)
    pk, _ = find_peaks(halus, height=nf + ambang_db, distance=max(1, int(150e3 / (f[1] - f[0]))))
    hasil = []
    for i in pk:
        f0 = f[i]
        if fc:   # tepatkan ke raster 200 kHz bila frekuensi absolut diketahui
            f0 = round((fc * 1e6 + f0) / 0.2e6) * 0.2e6 - fc * 1e6
        hasil.append((f0, halus[i] - nf))
    return hasil, nf


def fcch(x, fs, f0):
    """Awal burst FCCH (sampel pada fs/DESIMASI) dan frekuensi nada terukur pada carrier f0 (Hz relatif).
    Kembalikan (awal, nada_hz, rasio, kanal, fs_turun)."""
    bb = np.empty_like(x)
    for i in range(0, len(x), 1 << 20):              # geser frekuensi per blok: hemat memori
        n = np.arange(i, min(len(x), i + (1 << 20)))
        bb[i:i + len(n)] = x[i:i + len(n)] * np.exp(-2j * np.pi * f0 / fs * n)
    bb = resample_poly(bb, 1, DESIMASI).astype(np.complex64)
    fs = fs / DESIMASI
    n = np.arange(len(bb))
    kanal = lfilter(firwin(63, 90e3, fs=fs), 1, bb).astype(np.complex64)
    nada = lfilter(firwin(101, 8e3, fs=fs), 1, bb * np.exp(-2j * np.pi * F_FCCH / fs * n).astype(np.complex64))
    w = int(fs * 100e-6)
    rasio = np.convolve(np.abs(nada) ** 2, np.ones(w), "same") / (np.convolve(np.abs(kanal) ** 2, np.ones(w), "same") + 1e-9)
    on = rasio > 0.5
    d = np.diff(np.r_[0, on.astype(int), 0])
    naik, turun = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    # burst FCCH = 1 timeslot (577 us); nada yang lebih pendek dari 350 us bukan FCCH
    awal = [t for t, u in zip(naik, turun) if (u - t) >= fs * 350e-6]
    fr = []
    for t in awal:
        seg = kanal[t + int(fs * 100e-6):t + int(fs * 400e-6)]
        if len(seg) > 10:
            fr.append(np.angle(np.sum(seg[1:] * np.conj(seg[:-1]))) * fs / (2 * np.pi))
    return np.array(awal), (np.median(fr) if fr else np.nan), rasio.astype(np.float32), kanal, fs


def main():
    p = argparse.ArgumentParser(description="Bentuk sinyal GSM: carrier 200 kHz dan FCCH")
    p.add_argument("rekaman")
    p.add_argument("--fs", type=float, default=4e6)
    p.add_argument("--fc", type=float, default=None, help="frekuensi tengah rekaman (MHz)")
    p.add_argument("--detik", type=float, default=None, help="batasi panjang yang dianalisis")
    p.add_argument("--out", default="gambar/p6_gsm")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    x, klip = baca_cs8(a.rekaman, a.fs, maks_s=a.detik)
    cek_klip(klip, a.rekaman)
    f, psd = welch(x[:int(a.fs * 0.5)], a.fs, nperseg=8192, return_onesided=False)
    o = np.argsort(f)
    f, p_db = f[o], 10 * np.log10(psd[o] + 1e-12)
    m = np.abs(f) < 0.45 * a.fs
    carrier, nf = cari_carrier(f[m], p_db[m], a.fc)
    print("%s: %s detik, %d carrier sempit di atas noise" % (a.rekaman, koma("%.1f" % (len(x) / a.fs)),
                                                            len(carrier)))

    hasil = []
    for f0, lvl in carrier:
        awal, nada, _, _, fsd = fcch(x, a.fs, f0)
        ab = (a.fc * 1e6 + f0) / 1e6 if a.fc else None
        hasil.append(dict(f0=f0, lvl=lvl, awal=awal, nada=nada, abs=ab, arfcn=arfcn(ab) if ab else None))
        teks = "  %s MHz%s  +%.0f dB  FCCH %d kali" % (
            koma("%.1f" % ab) if ab else koma("%+.2f" % (f0 / 1e6)),
            ("  ARFCN %s" % hasil[-1]["arfcn"]) if hasil[-1]["arfcn"] is not None else "", lvl, len(awal))
        print(teks)

    bcch = [h for h in hasil if len(h["awal"]) >= 5]
    if not bcch:
        print("Tidak ada FCCH: carrier BCCH tidak ada di rekaman ini. Geser frekuensi tengah.")
        return
    b = max(bcch, key=lambda h: h["lvl"])
    _, _, b["rasio"], b["kanal"], fsd = fcch(x, a.fs, b["f0"])     # hitung ulang hanya untuk carrier terpilih
    sel = np.diff(b["awal"]) / fsd * 1e3
    frame = sel / FRAME_MS
    print("\nCarrier BCCH %s: selang FCCH (ms) %s" % (koma("%.1f" % b["abs"]) if b["abs"] else "",
                                                     koma(" ".join("%.2f" % s for s in sel[:12]))))
    print("  = %s frame TDMA" % " ".join("%.1f" % v for v in frame[:12]).replace(".", ","))
    print("  seluruh rekaman: %d selang, %d x 10 frame, %d x 11 frame, %d lain (FCCH terlewat)"
          % (len(frame), np.sum(np.abs(frame - 10) < 0.2), np.sum(np.abs(frame - 11) < 0.2),
             np.sum((np.abs(frame - 10) >= 0.2) & (np.abs(frame - 11) >= 0.2))))
    sepuluh = sel[np.abs(frame - 10) < 0.2]
    if len(sepuluh):
        print("  panjang frame TDMA terukur = %s ms (teori 4,615 ms), 1 timeslot = %s us"
              % (koma("%.3f" % (np.mean(sepuluh) / 10)), koma("%.1f" % (np.mean(sepuluh) / 10 / 8 * 1000))))
    cfo = b["nada"] - F_FCCH
    if a.fc:
        print("  nada FCCH terukur %s kHz -> CFO %+.0f Hz = %s ppm" % (koma("%.2f" % (b["nada"] / 1e3)), cfo,
                                                                    koma("%+.1f" % (cfo / (a.fc * 1e6 + b["f0"]) * 1e6))))

    # ------------------------------------------------------------- grafik
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.8, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.plot(f / 1e6, p_db - nf, color=GELAP, lw=0.7)
    for h in hasil:
        warna = ORANYE if len(h["awal"]) >= 5 else TINTA2
        lbl = ("ARFCN %d" % h["arfcn"]) if h["arfcn"] is not None else koma("%+.1f" % (h["f0"] / 1e6))
        ax.annotate(lbl + ("\nBCCH" if len(h["awal"]) >= 5 else ""), (h["f0"] / 1e6, h["lvl"] + 1),
                    ha="center", fontsize=FONT - 3, color=warna, fontweight="bold")
    ax.set_xlabel("frekuensi relatif terhadap %s MHz (MHz)" % koma("%g" % a.fc) if a.fc else "frekuensi (MHz)")
    ax.set_ylabel("dB di atas noise")
    ax.set_ylim(-5, max(h["lvl"] for h in hasil) + 10)
    judul(ax, "Carrier GSM: masing-masing 200 kHz (bandingkan LTE: 9-18 MHz dalam satu blok)")
    gaya(ax)

    ax = fig.add_subplot(gs[1])
    tampil = min(len(b["rasio"]), int(fsd * 1.0))                    # detik pertama
    t = np.arange(tampil) / fsd * 1e3
    s = max(1, int(fsd * 50e-6))
    ax.plot(t[::s], b["rasio"][:tampil:s], color=GELAP, lw=0.7)
    for i0, i1 in zip(b["awal"][:-1], b["awal"][1:]):
        if i1 >= tampil:
            break
        xm = (i0 + i1) / 2 / fsd * 1e3
        ax.annotate("", xy=(i1 / fsd * 1e3, 1.05), xytext=(i0 / fsd * 1e3, 1.05),
                    arrowprops=dict(arrowstyle="<->", color=ORANYE, lw=1.2))
        ax.text(xm, 1.1, koma("%.1f" % ((i1 - i0) / fsd * 1e3)), ha="center", fontsize=FONT - 4, color=TINTA)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("waktu (ms)")
    ax.set_ylabel("porsi nada +67,7 kHz")
    judul(ax, "FCCH di carrier BCCH: selang 10-10-10-10-11 frame (46,2 / 50,8 ms) = multiframe 51")
    gaya(ax)

    ax = fig.add_subplot(gs[2])
    t0 = int(b["awal"][1]) - int(fsd * 150e-6)
    n = int(fsd * FRAME_MS * 1.25e-3)
    w = max(1, int(fsd * 4e-6))
    seg = np.abs(b["kanal"][t0:t0 + n]) ** 2
    pw = 10 * np.log10(np.convolve(seg, np.ones(w) / w, "same") + 1e-6)
    naik = int(np.argmin(pw[:int(fsd * 300e-6)]))           # celah guard tepat sebelum burst FCCH
    pw = pw[naik:]
    tt = np.arange(len(pw)) / fsd * 1e3
    ax.plot(tt, pw, color=GELAP, lw=0.6)
    for k in range(11):
        x0 = k * TS_US / 1000
        ax.axvline(x0, color=BIRU, lw=0.8, alpha=0.6)
        if k < 10:
            ax.text(x0 + TS_US / 2000, pw.max() + 1, "TS%d" % (k % 8), ha="center", fontsize=FONT - 3,
                    color=ORANYE if k % 8 == 0 else TINTA2, fontweight="bold")
    ax.axvspan(0, TS_US / 1000, color=ORANYE, alpha=0.12, lw=0)
    ax.set_ylim(pw.min(), pw.max() + 3)
    ax.set_xlabel("waktu sejak burst FCCH (ms)  -  1 frame TDMA = 8 timeslot x 577 us = 4,615 ms")
    ax.set_ylabel("daya (dB)")
    judul(ax, "Satu frame TDMA di carrier BCCH: TS0 = FCCH (nada murni, arsir); celah antar-burst tiap 577 us")
    gaya(ax)
    fig.savefig(a.out + "_gsm.png", dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Selesai: %s_gsm.png" % a.out)


if __name__ == "__main__":
    main()
