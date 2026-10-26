#!/usr/bin/env python3
"""
bentuk_sinyal.py - Percobaan 3 & 5: bentuk sinyal LTE (OFDMA) dari rekaman IQ HackRF (.cs8).

Rekam satu carrier tepat di frekuensi tengahnya (contoh: LTE 20 MHz B3 di 1870 MHz, 1 detik):
  hackrf_transfer -r data/lte1870.cs8 -f 1870000000 -s 20000000 -b 20000000 -n 20000000 \
      -l 32 -g 24 -a 0
  # TDD B40:
  hackrf_transfer -r data/tdd2330.cs8 -f 2330000000 -s 20000000 -b 20000000 -n 10000000 \
      -l 40 -g 30 -a 0

Analisis:
  python3 bentuk_sinyal.py data/lte1870.cs8 --fc 1870 --out gambar/p3_lte1870
  python3 bentuk_sinyal.py data/tdd2330.cs8 --fc 2330 --out gambar/p5_tdd2330

Keluaran (prefix --out):
  _spektrum.png     PSD: lebar terisi -> jumlah resource block (RB = 12 x 15 kHz = 180 kHz)
  _spektrogram.png  waktu x frekuensi beberapa ms, garis tiap 1 ms (subframe)
  _ofdm.png         autokorelasi cyclic prefix: durasi simbol T_u -> spasi subcarrier 1/T_u,
                    dan metrik CP 1 ms: satu puncak = satu simbol OFDM
  _selubung.png     daya terhadap waktu: FDD downlink selalu menyala, TDD on/off tiap 5/10 ms

Rekaman harus bebas clipping: skrip memberi peringatan bila ada sampel mentok +-127.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch

from seluler_dasar import (BIRU, FONT, GELAP, ORANYE, TINTA, TINTA2, baca_cs8, bw_lte, cek_klip, gaya,
                           judul, koma, spektrogram)


def lebar_terisi(f, p_db, di_atas_noise_db=2.5):
    """Tepi carrier yang berpusat di DC: dari +-0,3 MHz berjalan keluar selama PSD (dihaluskan 100 kHz)
    masih di_atas_noise_db di atas noise floor. Puncak sempit di luar tepi (GSM, spur) tidak terhitung."""
    df = f[1] - f[0]
    k = max(1, int(100e3 / df))
    halus = np.convolve(p_db, np.ones(k) / k, "same")
    plateau = np.median(p_db[(np.abs(f) < 1.5e6) & (np.abs(f) > 50e3)])
    on = halus >= np.percentile(halus[k:-k], 3) + di_atas_noise_db
    i0 = np.argmin(np.abs(f))
    lo = i0 - int(0.3e6 / df)
    while lo > 0 and on[lo - 1]:
        lo -= 1
    hi = i0 + int(0.3e6 / df)
    while hi < len(f) - 1 and on[hi + 1]:
        hi += 1
    return f[lo], f[hi], plateau


def autokorelasi(x, fs, lo_hz, hi_hz):
    """|R(lag)| ternormalisasi untuk lag 0..120 us; spektrum di luar carrier dibuang dan diratakan
    (whitening) supaya bentuk spektrum dan spur tidak mendominasi."""
    n = 1 << int(np.log2(min(len(x), int(fs * 0.2))))
    X = np.fft.fft(x[:n])
    f = np.fft.fftfreq(n, 1 / fs)
    m = (f > lo_hz) & (f < hi_hz) & (np.abs(f) > 20e3)
    P = np.convolve(np.abs(X) ** 2, np.ones(2001) / 2001, "same")
    Xw = np.where(m, X / np.sqrt(P + 1e-9), 0)
    R = np.fft.ifft(np.abs(Xw) ** 2)[:int(fs * 120e-6)]
    return np.abs(R) / R[0].real


def metrik_cp(x, fs, n_u, n_cp, ms=50):
    """M(n) = |sum_{k<n_cp} x*(n+k) x(n+k+n_u)| memuncak di awal setiap simbol OFDM. Dirata-rata
    per slot 0,5 ms selama `ms` ms (semua slot punya pola 7 simbol yang sama), dikembalikan 1 ms."""
    slot = int(round(fs * 0.5e-3))
    n = int(fs * ms / 1000) // slot * slot
    prod = np.conj(x[:n + n_cp]) * x[n_u:n_u + n + n_cp]
    c = np.cumsum(np.r_[0, prod])
    M = np.abs(c[n_cp:n_cp + n] - c[:n])
    rata = M.reshape(-1, slot).mean(axis=0)
    return np.r_[rata, rata]


def grafik_spektrum(f, p_db, lo, hi, plateau, fc, path):
    lebar = (hi - lo) / 1e6
    nrb, bw = bw_lte(lebar)
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.plot(f / 1e6, p_db - plateau, color=GELAP, lw=0.8)
    ax.axvspan(lo / 1e6, hi / 1e6, color=ORANYE, alpha=0.12, lw=0)
    y = 4
    ax.annotate("", xy=(hi / 1e6, y), xytext=(lo / 1e6, y), arrowprops=dict(arrowstyle="<->", color=ORANYE, lw=2))
    ax.text((lo + hi) / 2e6, y + 1, koma("lebar terisi %.2f MHz  ~  %d RB x 180 kHz  ->  kanal LTE %g MHz"
                                         % (lebar, nrb, bw)), ha="center", va="bottom", fontsize=FONT,
            color=TINTA, fontweight="bold")
    ax.set_ylim(-35, 12)
    ax.set_xlim(f[0] / 1e6, f[-1] / 1e6)
    ax.set_xlabel("frekuensi relatif terhadap %s MHz (MHz)" % koma("%g" % fc) if fc else "frekuensi (MHz)")
    ax.set_ylabel("dB relatif plateau")
    judul(ax, "Spektrum rata-rata: blok rata (OFDM) dengan tepi tajam")
    gaya(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def grafik_spektrogram(x, fs, ms, fc, path):
    nfft = 256
    seg = x[:int(fs * ms / 1000)]
    S = spektrogram(seg, nfft)
    v0 = np.percentile(S, 20)
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.imshow(S, aspect="auto", origin="lower", cmap="magma", vmin=v0, vmax=v0 + 30,
              extent=[0, len(seg) / fs * 1e3, -fs / 2e6, fs / 2e6], interpolation="nearest")
    for k in range(int(ms) + 1):
        ax.axvline(k, color="white", lw=0.6, alpha=0.6)
    ax.set_xlabel("waktu (ms)  -  garis putih tiap 1 ms = 1 subframe")
    ax.set_ylabel("frekuensi relatif (MHz)")
    judul(ax, koma("Spektrogram %g ms%s: resolusi %.0f kHz x %.1f us"
                   % (ms, (" di %g MHz" % fc) if fc else "", fs / nfft / 1e3, nfft / fs * 1e6)))
    gaya(ax, grid_y=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def grafik_ofdm(R, fs, t_u, M, n_u, path):
    lag_us = np.arange(len(R)) / fs * 1e6
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    m = lag_us > 10
    a.plot(lag_us[m], R[m], color=GELAP, lw=0.9)
    i = int(round(t_u * fs))
    i = i - 1 + int(np.argmax(R[i - 1:i + 2]))
    a.plot(lag_us[i], R[i], "o", color=ORANYE, ms=9)
    a.annotate(koma("T_u = %.2f us\n1/T_u = %.2f kHz" % (t_u * 1e6, 1e-3 / t_u)), (lag_us[i], R[i]),
               xytext=(15, -10), textcoords="offset points", fontsize=FONT, color=TINTA, fontweight="bold",
               va="top")
    a.set_xlabel("lag (us)")
    a.set_ylabel("|autokorelasi|")
    judul(a, "Cyclic prefix = salinan ekor simbol")
    gaya(a)

    a = ax[1]
    t = np.arange(len(M)) / fs * 1e3
    a.plot(t, M / M.max(), color=GELAP, lw=0.8)
    a.axvline(0.5, color=ORANYE, lw=1.2, ls="--") if t[-1] > 0.5 else None
    a.set_xlabel("waktu (ms)")
    a.set_ylabel("metrik CP")
    judul(a, "Metrik CP, rata-rata 100 slot: 7 simbol per slot 0,5 ms")
    gaya(a)
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def selubung(x, fs, ms, res_us=10):
    w = int(fs * res_us * 1e-6)
    seg = x[:int(fs * ms / 1000) // w * w]
    p = 10 * np.log10(np.mean(np.abs(seg.reshape(-1, w)) ** 2, axis=1) + 1e-6)
    lo, hi = np.percentile(p, 5), np.percentile(p, 95)
    ambang = (lo + hi) / 2
    # jeda TDD (slot uplink) >= 0,5 ms; jeda antar simbol CRS pada sel FDD sepi hanya ~0,2 ms
    on = np.convolve(p > ambang, np.ones(3), "same") > 0
    tepi = np.flatnonzero(np.diff(np.r_[1, on.astype(int), 1]))
    mulai, akhir = tepi[0::2], tepi[1::2]
    panjang = (akhir - mulai) * res_us >= 500
    jeda = mulai[panjang]
    arsir = np.zeros(len(p), bool)
    for m0, m1 in zip(mulai[panjang], akhir[panjang]):
        arsir[m0:m1] = True
    periode = None
    if len(jeda) >= 3:
        selang = np.diff(jeda) * res_us / 1000
        if np.std(selang) < 0.3:
            periode = float(np.median(selang))
    return p, ambang, on, arsir, periode


def grafik_selubung(p, ambang, on, arsir, periode, res_us, path):
    t = np.arange(len(p)) * res_us / 1000
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.plot(t, p, color=GELAP, lw=0.6)
    if periode:
        ax.fill_between(t, p.min(), p.max(), where=arsir, color=BIRU, alpha=0.15, lw=0, step="mid")
    ax.axhline(ambang, color=TINTA2, ls="--", lw=1)
    if periode:
        teks = koma("TDD: pancaran berhenti berkala (arsir biru = slot uplink/guard), periode %.1f ms, "
                    "downlink %.0f%% waktu" % (periode, 100 * (1 - arsir.mean())))
    else:
        teks = "Downlink menyala terus (FDD): tidak ada jeda berkala"
    judul(ax, teks)
    ax.set_xlabel("waktu (ms)")
    ax.set_ylabel(koma("daya per %g us (dBFS)" % res_us))
    ax.set_xlim(0, t[-1])
    gaya(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Bentuk sinyal LTE dari rekaman .cs8")
    p.add_argument("rekaman")
    p.add_argument("--fs", type=float, default=20e6, help="sample rate rekaman (default 20e6)")
    p.add_argument("--fc", type=float, default=None, help="frekuensi tengah rekaman (MHz), untuk label")
    p.add_argument("--ms", type=float, default=12, help="panjang spektrogram (ms)")
    p.add_argument("--ms-selubung", type=float, default=40)
    p.add_argument("--out", default="gambar/p3")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    x, klip = baca_cs8(a.rekaman, a.fs, maks_s=0.5)
    cek_klip(klip, a.rekaman)
    print("%s: %s ms, %s Msps" % (a.rekaman, koma("%.0f" % (len(x) / a.fs * 1e3)), koma("%g" % (a.fs / 1e6))))

    f, psd = welch(x[:int(a.fs * 0.2)], a.fs, nperseg=4096, return_onesided=False)
    o = np.argsort(f)
    f, p_db = f[o], 10 * np.log10(psd[o] + 1e-12)
    lo, hi, plateau = lebar_terisi(f, p_db)
    nrb, bw = bw_lte((hi - lo) / 1e6)
    print("1. Spektrum : lebar terisi %s MHz  ->  %d RB x 180 kHz  ->  kanal LTE %g MHz"
          % (koma("%.2f" % ((hi - lo) / 1e6)), nrb, bw))
    grafik_spektrum(f, p_db, lo, hi, plateau, a.fc, a.out + "_spektrum.png")
    grafik_spektrogram(x, a.fs, a.ms, a.fc, a.out + "_spektrogram.png")

    R = autokorelasi(x, a.fs, lo, hi)
    # sel yang sepi hampir hanya berisi CRS (tiap 3 subcarrier) -> puncak palsu di T_u/3 dan 2T_u/3;
    # CP memberi puncak di T_u, jadi ambil lag terbesar di antara puncak yang >= 50% puncak tertinggi
    lag0 = int(a.fs * 20e-6)
    pk, _ = find_peaks(R[lag0:], distance=int(a.fs * 2e-6))
    pk += lag0
    n_u = int(pk[R[pk] >= 0.5 * R[pk].max()].max())
    # interpolasi parabola di sekitar puncak (lag pecahan, mis. 1333,3 sampel di 20 Msps)
    y0, y1, y2 = R[n_u - 1], R[n_u], R[n_u + 1]
    t_u = (n_u + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) / a.fs
    print("2. OFDM     : puncak autokorelasi di lag %d sampel = %s us  ->  spasi subcarrier 1/T_u = %s kHz"
          % (n_u, koma("%.2f" % (t_u * 1e6)), koma("%.2f" % (1e-3 / t_u))))
    print("              puncak/median = %.0fx (makin besar makin yakin ini OFDM)"
          % (R[n_u] / np.median(R[lag0:])))
    M = metrik_cp(x, a.fs, n_u, max(4, int(round(4.69e-6 * a.fs))))
    grafik_ofdm(R, a.fs, t_u, M, n_u, a.out + "_ofdm.png")

    pw, ambang, on, arsir, periode = selubung(x, a.fs, a.ms_selubung)
    if periode:
        print("3. Selubung : TDD, pola on/off berulang tiap %s ms, downlink %.0f%% waktu"
              % (koma("%.1f" % periode), 100 * (1 - arsir.mean())))
    else:
        print("3. Selubung : menyala terus -> downlink FDD")
    grafik_selubung(pw, ambang, on, arsir, periode, 10, a.out + "_selubung.png")
    print("Selesai: %s_*.png" % a.out)


if __name__ == "__main__":
    main()
