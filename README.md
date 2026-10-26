# Praktikum Sinyal Seluler: paket mahasiswa

CIF65217 Jaringan Nirkabel · Bab 5 Wireless WAN: Seluler

Paket ini berisi materi, skrip analisis, dan rekaman sinyal seluler yang direkam dosen dengan HackRF One saat demonstrasi. **HackRF tidak diperlukan**: semua percobaan di bawah dianalisis dari rekaman.

## 1. Buka materi

Buka `materi_seluler.html` di browser (bisa offline). Isinya teori Bab 5 dan penjelasan tiap percobaan. Pertanyaan yang harus dijawab ada di `praktikum_seluler/PANDUAN.md`.

**Take-home:** buka `kasus_detektif_seluler.html` untuk tiga kasus "Detektif Seluler" (dibahas di pertemuan berikutnya). Rekaman Kasus 2 ada di `praktikum_seluler/data/misteri/`.

## 2. Pasang Python

Python 3.8 atau lebih baru, lalu dari folder ini:

```bash
pip install -r requirements.txt
```

Di Windows, ganti `python3` dengan `python` pada semua perintah.

## 3. Struktur folder

```
untuk_mahasiswa/
├── MULAI_DI_SINI.md          ← file ini
├── materi_seluler.html       ← teori + praktikum
├── kasus_detektif_seluler.html ← take-home: tiga kasus untuk pertemuan berikutnya
├── requirements.txt
├── lte_cell_search/
│   └── lte_common.py         ← dekoder PSS/SSS/PBCH, dipakai cari_sel_lte.py
└── praktikum_seluler/        ← jalankan semua perintah dari folder INI
    ├── PANDUAN.md            ← langkah dan pertanyaan tiap percobaan
    ├── peta_seluler.py       ← P1, P2
    ├── bentuk_sinyal.py      ← P3, P5
    ├── cari_sel_lte.py       ← P4 (PCI + MIB)
    ├── gsm_fcch.py           ← P6
    ├── seluler_dasar.py
    ├── data/lokasi_X/        ← rekaman dosen (.cs8 = IQ 8 bit, .csv = sweep)
    ├── data/misteri/         ← lima rekaman tanpa label untuk Kasus 2
    └── contoh_hasil/lokasi_X/  ← grafik dan ringkasan hasil dosen, untuk pembanding
```

Folder `lte_cell_search/` harus tetap di sebelah `praktikum_seluler/`.

## 4. Jalankan analisis

```bash
cd praktikum_seluler
```

Perintah `hackrf_sweep` dan `hackrf_transfer` di PANDUAN.md adalah perintah perekaman yang dijalankan dosen. Kalian cukup menjalankan perintah analisis di bawah ini; nama berkas, frekuensi, sample rate, dan kesalahan osilator (`--ppm`) sudah disesuaikan dengan rekamannya. Grafik hasil masuk ke folder `hasil/`.

### Lokasi A

**P1 · peta spektrum** (60 sapuan, 700–2700 MHz)
```bash
python3 peta_seluler.py data/lokasi_A/sweep_seluler.csv --out hasil/A/p1
python3 peta_seluler.py data/lokasi_A/sweep_seluler.csv --out hasil/A/p1_b8b3 --pita B8,B3   # dua pita saja
```

**P2 · downlink vs uplink** (sweep 120 detik)
```bash
python3 peta_seluler.py data/lokasi_A/sweep_b3.csv --out hasil/A/p2 --waterfall
```

**P3 + P4 · lte1870** (LTE di 1870 MHz, 2 s, 20e6 sps)
```bash
python3 bentuk_sinyal.py data/lokasi_A/lte1870.cs8 --fs 20e6 --fc 1870 --out hasil/A/p3_lte1870
python3 cari_sel_lte.py data/lokasi_A/lte1870.cs8 --fs 20e6 --fc 1870 --ms 2000 --frame-mib 198 --ppm -1.23 --out hasil/A/p4_lte1870
```

**P3 + P4 · lte945.2** (LTE di 945.2 MHz, 1 s, 15.36e6 sps)
```bash
python3 bentuk_sinyal.py data/lokasi_A/lte945.2.cs8 --fs 15.36e6 --fc 945.2 --out hasil/A/p3_lte945.2
python3 cari_sel_lte.py data/lokasi_A/lte945.2.cs8 --fs 15.36e6 --fc 945.2 --ms 1000 --frame-mib 98 --ppm -1.23 --out hasil/A/p4_lte945.2
```

**P5 · TDD tdd2330.6** (pusat carrier 2329.8 MHz, rekaman di 2330.6 MHz)
```bash
python3 bentuk_sinyal.py data/lokasi_A/tdd2330.6.cs8 --fc 2330.6 --ms-selubung 100 --out hasil/A/p5_tdd2330.6
python3 cari_sel_lte.py data/lokasi_A/tdd2330.6.cs8 --fc 2329.8 --geser-khz -800 --ms 1000 --frame-mib 98 --cfo-maks-khz 12 --ppm -1.23 --out hasil/A/p5_tdd2330.6
```

**P6 · GSM gsm1859.6** (5 s di sekitar 1859.6 MHz)
```bash
python3 gsm_fcch.py data/lokasi_A/gsm1859.6.cs8 --fc 1859.6 --out hasil/A/p6_gsm1859.6
```

**P7 · survei sel**: survei butuh HackRF, jadi hasilnya sudah disertakan: `contoh_hasil/lokasi_A/survei_inventaris.csv`, `survei_inventaris.png`, dan `survei_catatan.txt`.

Bandingkan hasilmu dengan `contoh_hasil/lokasi_A/ringkasan.txt` (hasil dosen saat merekam).

## 5. Kalau ada masalah

| Gejala | Solusi |
|---|---|
| `ModuleNotFoundError: numpy` | `pip install -r requirements.txt` |
| `lte_common.py tidak ditemukan` | Pastikan `lte_cell_search/` ada di sebelah `praktikum_seluler/` |
| Laptop lambat atau kehabisan memori di P4 | Kecilkan `--ms` (mis. `--ms 500`) dan `--frame-mib` (mis. `--frame-mib 40`) |
| Hasil sedikit berbeda dari `contoh_hasil` | Wajar untuk angka seperti skor PSS; PCI, bandwidth, 15 kHz, dan 4,615 ms harus sama |
