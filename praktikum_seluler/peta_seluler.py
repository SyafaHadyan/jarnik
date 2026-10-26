#!/usr/bin/env python3
"""
peta_seluler.py - Percobaan 1 & 2: peta spektrum pita seluler dari hackrf_sweep (CSV).

Rekam (hanya menerima):
  hackrf_sweep -f 700:2700 -w 100000 -l 32 -g 30 -a 0 -N 30 -r data/sweep_seluler.csv
  # uplink B3 sambil ponsel di dekat antena mengunggah video:
  hackrf_sweep -f 1700:1890 -w 100000 -l 32 -g 30 -a 0 -N 300 -r data/sweep_b3.csv

Analisis:
  python3 peta_seluler.py data/sweep_seluler.csv --out gambar/p1
  python3 peta_seluler.py data/sweep_b3.csv --out gambar/p2 --waterfall
  python3 peta_seluler.py data/sweep_seluler.csv --out gambar/p1_b8b3 --pita B8,B3   # panel tertentu

Keluaran (prefix --out):
  _peta.png       spektrum rata-rata + max-hold per pita, arsir DL (oranye) / UL (biru)
  _carrier.csv    daftar carrier: frekuensi tengah, lebar, jenis, pita, EARFCN, tebakan BW LTE
  _waterfall.png  (dengan --waterfall) waktu x frekuensi, untuk melihat burst uplink

Aturan deteksi (sengaja sederhana supaya bisa dicek dengan tangan):
  noise floor = persentil 10 rata-rata; bin "terisi" bila rata-rata >= noise + --ambang-db.
  Bin terisi yang bersebelahan (celah <= 0,3 MHz) digabung menjadi satu carrier.
  Lebar >= 3 MHz -> carrier lebar (LTE/NR); lebar <= 0,4 MHz -> sempit (GSM 200 kHz atau spur).
  Carrier lebar dengan puncak sempit > 12 dB di atas medianya ditandai "+ puncak sempit": biasanya
  ada carrier GSM di sebelahnya yang ikut tergabung. Tabel ini tebakan kasar; pastikan dengan
  rekaman IQ (Percobaan 3).
  Spur HackRF biasanya tepat di kelipatan MHz bulat dan tetap ada saat antena dilepas.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seluler_dasar import (BIRU, FONT, GELAP, ORANYE, PITA, TINTA, TINTA2, baca_sweep, bw_lte,
                           earfcn, gaya, judul, koma, pita_dari_frek)


def cari_carrier(f, avg, nf, ambang_db, celah_mhz=0.3):
    step = np.median(np.diff(f))
    on = avg >= nf + ambang_db
    hasil, i = [], 0
    while i < len(f):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(f) and (on[j + 1] or np.any(on[j + 1:j + 1 + int(celah_mhz / step)])):
            j += 1
        lo, hi = f[i] - step / 2, f[j] + step / 2
        seg = avg[i:j + 1]
        hasil.append(dict(lo=lo, hi=hi, lebar=hi - lo, tengah=(lo + hi) / 2,
                          puncak_db=seg.max() - nf, rata_db=np.median(seg) - nf))
        i = j + 1
    return hasil


LEBAR_MIN = 3.0


def jenis(c):
    if c["lebar"] >= LEBAR_MIN:
        nrb, bw = bw_lte(c["lebar"])
        j = "lebar + puncak sempit" if c["puncak_db"] - c["rata_db"] > 12 else "lebar (LTE/NR)"
        return j, "%d RB ~ %s MHz" % (nrb, koma("%g" % bw))
    if c["lebar"] <= 0.4:
        bulat = abs(c["tengah"] - round(c["tengah"])) < 0.15
        return ("sempit: spur?" if bulat else "sempit (GSM?)"), ""
    return "sedang", ""


def grafik_peta(f, avg, mx, nf, carrier, path, pilih=None):
    panel = [b for b in PITA if any(c["lebar"] >= LEBAR_MIN and b[2][0] <= c["tengah"] <= b[2][1]
                                    for c in carrier)]
    if pilih:
        panel = [b for b in panel if b[0].split()[0] in pilih]
    if not panel:
        panel = [("rentang", "", (f[0], f[-1]), None)]
    fig, ax = plt.subplots(len(panel), 1, figsize=(16, 3.0 * len(panel) + 0.6))
    ax = np.atleast_1d(ax)
    for a, b in zip(ax, panel):
        nama, mode, dl, ul = b[:4]
        rng = [dl] + ([ul] if ul else [])
        lo = max(f[0], min(r[0] for r in rng) - 5)
        hi = min(f[-1], max(r[1] for r in rng) + 5)
        m = (f >= lo) & (f <= hi)
        a.axvspan(*dl, color=ORANYE, alpha=0.10, lw=0)
        a.text(dl[0], 0.97, " %s %s" % (nama, "TDD (DL+UL)" if mode == "TDD" else "downlink"),
               transform=a.get_xaxis_transform(), va="top", fontsize=FONT - 2, color=ORANYE, fontweight="bold")
        if ul:
            a.axvspan(*ul, color=BIRU, alpha=0.08, lw=0)
            a.text(ul[0], 0.97, " uplink", transform=a.get_xaxis_transform(), va="top",
                   fontsize=FONT - 2, color=BIRU, fontweight="bold")
        a.plot(f[m], mx[m] - nf, color="#B9C2CA", lw=0.6, label="max-hold")
        a.plot(f[m], avg[m] - nf, color=GELAP, lw=0.9, label="rata-rata")
        for c in carrier:
            if c["lebar"] >= LEBAR_MIN and lo <= c["tengah"] <= hi:
                a.annotate(koma("%.1f MHz" % c["lebar"]), (c["tengah"], c["rata_db"] + 3), ha="center",
                           fontsize=FONT - 4, color=TINTA)
        a.set_xlim(lo, hi)
        a.set_ylim(-3, max(10, (mx[m] - nf).max() + 4))
        a.set_ylabel("dB di atas noise")
        gaya(a)
    ax[0].legend(loc="upper right", fontsize=FONT - 3, frameon=False)
    ax[-1].set_xlabel("frekuensi (MHz)")
    fig.tight_layout(h_pad=1.0)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def grafik_waterfall(f, m, nf, path):
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(m - nf, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=35,
                   extent=[f[0], f[-1], 0, m.shape[0]], interpolation="nearest")
    for b in PITA:
        for rng, w, lbl in [(b[2], ORANYE, "DL"), (b[3], BIRU, "UL")]:
            if rng and rng[1] > f[0] and rng[0] < f[-1]:
                ax.annotate("", xy=(min(rng[1], f[-1]), m.shape[0] * 1.02),
                            xytext=(max(rng[0], f[0]), m.shape[0] * 1.02), annotation_clip=False,
                            arrowprops=dict(arrowstyle="<->", color=w, lw=2))
                ax.text((max(rng[0], f[0]) + min(rng[1], f[-1])) / 2, m.shape[0] * 1.04,
                        "%s %s" % (b[0], lbl), ha="center", color=w, fontsize=FONT - 1, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("dB di atas noise", color=TINTA2)
    ax.set_xlabel("frekuensi (MHz)")
    ax.set_ylabel("nomor sapuan (waktu ->)")
    gaya(ax, grid_y=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Peta spektrum seluler dari CSV hackrf_sweep")
    p.add_argument("sweep")
    p.add_argument("--out", default="gambar/p1")
    p.add_argument("--ambang-db", type=float, default=6.0)
    p.add_argument("--waterfall", action="store_true")
    p.add_argument("--pita", default=None, help="hanya panel ini, mis. B8,B3")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    f, m = baca_sweep(a.sweep)
    avg, mx = 10 * np.log10(np.mean(10 ** (m / 10), axis=0)), m.max(axis=0)
    nf = np.percentile(avg, 10)
    carrier = cari_carrier(f, avg, nf, a.ambang_db)
    print("%d sapuan, %s - %s MHz, noise floor %.1f dB" % (m.shape[0], koma("%g" % f[0]), koma("%g" % f[-1]), nf))

    with open(a.out + "_carrier.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tengah_mhz", "lebar_mhz", "jenis", "tebakan_bw", "pita", "arah", "earfcn",
                    "rata_db", "puncak_db"])
        print("\n%-8s %-6s %-22s %-16s %-9s %-4s %-6s %s" % ("tengah", "lebar", "jenis", "tebakan",
                                                            "pita", "arah", "EARFCN", "dB"))
        for c in carrier:
            j, tebak = jenis(c)
            pita, arah, _ = pita_dari_frek(c["tengah"])
            ea = earfcn(round(c["tengah"], 1)) if c["lebar"] >= LEBAR_MIN else None
            w.writerow(["%.1f" % c["tengah"], "%.1f" % c["lebar"], j, tebak, pita or "", arah or "",
                        ea if ea is not None else "", "%.1f" % c["rata_db"], "%.1f" % c["puncak_db"]])
            if c["lebar"] >= LEBAR_MIN and arah in ("DL", "TDD"):
                print("%-8s %-6s %-22s %-16s %-9s %-4s %-6s +%.0f" % (
                    "%.1f" % c["tengah"], "%.1f" % c["lebar"], j, tebak, pita or "-", arah or "-",
                    ea if ea is not None else "-", c["rata_db"]))

    sempit = [c for c in carrier if c["lebar"] <= 0.4]
    print("\n+ %d garis sempit (GSM atau spur, lihat _carrier.csv); %d di antaranya tepat di MHz bulat"
          % (len(sempit), sum(jenis(c)[0].endswith("spur?") for c in sempit)))
    ul = [c for c in carrier if c["lebar"] >= LEBAR_MIN and pita_dari_frek(c["tengah"])[1] == "UL"]
    print("  carrier lebar di pita uplink: %d" % len(ul))

    grafik_peta(f, avg, mx, nf, carrier, a.out + "_peta.png", a.pita.split(",") if a.pita else None)
    if a.waterfall:
        grafik_waterfall(f, m, nf, a.out + "_waterfall.png")
    print("\nSelesai: %s_*" % a.out)


if __name__ == "__main__":
    main()
