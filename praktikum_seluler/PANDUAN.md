# Praktikum: Melihat Sinyal Seluler dengan SDR

CIF65217 Jaringan Nirkabel · Bab 5 Wireless WAN: Seluler · HackRF One (hanya menerima)

Slide teori (Bab 5) menjelaskan sel, LTE, OFDMA, frame 10 ms, dan resource block 180 kHz. Di praktikum ini besaran-besaran itu diukur dari sinyal BTS/eNodeB di sekitar kampus.

**Format: demonstrasi terpandu.** Hanya ada satu HackRF. Dosen menjalankan perintah perekaman (`hackrf_sweep`, `hackrf_transfer`, `survei_sel.py`) di depan kelas dan menampilkan hasilnya. Mahasiswa mengikuti tiap langkah, lalu menjalankan skrip analisis yang sama di laptop sendiri pada contoh data di `data/` (atau rekaman yang dibagikan dosen setelah demonstrasi), dan menjawab pertanyaan di tiap percobaan.

| Konsep di slide | Yang diukur di praktikum | Percobaan |
|---|---|---|
| Keterbatasan spektrum, pita frekuensi (5.1) | Peta pita 700–2700 MHz: operator, pita, lebar kanal | 1 |
| Downlink vs uplink, FDD | DL selalu menyala, UL hanya burst saat ponsel memancar | 2 |
| OFDMA, 12 subcarrier × 15 kHz = 180 kHz (5.4) | Lebar kanal = N_RB × 180 kHz; durasi simbol 66,7 µs → 15 kHz | 3 |
| Sel, frequency reuse, eNodeB (5.1, 5.3) | Physical Cell ID dari PSS/SSS; beberapa sel di frekuensi yang sama | 4 |
| MIMO, multi antena (5.5) | Jumlah port antena eNodeB dari MIB (PBCH) | 4 |
| Frame 10 ms, subframe 1 ms, slot 0,5 ms (5.4) | PSS tiap 5 ms, 7 simbol per slot, TDD on/off tiap 5 ms | 3, 4, 5 |
| GSM generasi 2 (5.1) | Carrier 200 kHz, frame TDMA 4,615 ms, 8 timeslot | 6 |
| Semua di atas: sel, eNodeB, operator | Inventaris sel di sekitar lab dan eNodeB yang memancarkannya | 7 |

## Tujuan

Setelah praktikum, mahasiswa dapat:
1. Membaca spektrum dan spektrogram, serta membedakan sinyal asli dari spur internal SDR.
2. Mengidentifikasi pita seluler Indonesia (B1, B3, B5, B8, B40) dan lebar kanal LTE dari pengukuran.
3. Menunjukkan dengan data bahwa LTE downlink memakai OFDM dengan spasi subcarrier 15 kHz.
4. Menemukan Physical Cell ID (PCI) sel di sekitar dan mencocokkannya dengan ponsel.
5. Membedakan FDD dan TDD, serta GSM (TDMA, 200 kHz) dan LTE (OFDMA, 9–18 MHz).

## Alat dan bahan

- Satu HackRF One + antena untuk 700–2700 MHz, dioperasikan dosen (antena "LTE/4G" rubber-duck cocok; antena teleskopik bisa dipakai bila dipendekkan)
- Laptop dosen dengan `hackrf_transfer`, `hackrf_sweep` (paket `hackrf`) dan Python 3.8+ dengan `numpy scipy matplotlib`, tersambung ke proyektor
- Laptop mahasiswa: cukup Python 3.8+ dengan `numpy scipy matplotlib`, tanpa HackRF
- Ponsel Android dengan aplikasi info sel (misalnya *NetMonster* atau *Network Cell Info Lite*) untuk membaca PCI dan EARFCN
- Folder `praktikum_seluler/` dan `lte_cell_search/` beserta contoh data `data/` (±90 MB) sudah disalin ke laptop mahasiswa sebelum kelas

## Aturan (wajib dibaca)

1. **Hanya menerima.** HackRF bisa memancar. Jangan pernah memakai `hackrf_transfer -t`, `-x`, atau aplikasi pemancar. Memancar di pita seluler tanpa izin melanggar UU No. 36/1999 tentang Telekomunikasi dan mengganggu layanan publik.
2. **Tidak mendekode data pengguna.** Praktikum ini hanya mengamati bentuk sinyal dan sinyal broadcast yang memang dipancarkan untuk semua ponsel (PSS, SSS, FCCH). Penyadapan informasi dilarang oleh UU 36/1999 Pasal 40.
3. Rekaman IQ disimpan untuk keperluan praktikum saja dan dihapus setelah laporan dinilai.

## Persiapan (10 menit)

```bash
hackrf_info                      # laptop dosen: harus muncul "Found HackRF"
cd praktikum_seluler
python3 -c "import numpy, scipy, matplotlib; print('ok')"
```

**Gain dan clipping.** HackRF hanya 8 bit (−128..127). Bila gain terlalu tinggi, sinyal kuat "mentok" dan bentuknya rusak. Semua skrip memberi peringatan `PERINGATAN ... clipping` bila ini terjadi. Turunkan `-l` (LNA, 0–40 dB, langkah 8) atau `-g` (VGA, 0–62 dB, langkah 2), lalu rekam ulang. Amplifier RF `-a` selalu 0. Titik awal yang baik: B1/B3 `-l 32 -g 24`, B8 `-l 16 -g 16`, B40 `-l 40 -g 30`.

Contoh data dari pengukuran persiapan ada di `data/` dan contoh gambar di `gambar/`. Mahasiswa memakai data ini untuk mengulang analisis di laptop masing-masing: semua skrip kecuali `survei_sel.py` bisa berjalan tanpa HackRF. Perintah `hackrf_*` di tiap percobaan dijalankan dosen.

---

## Percobaan 1 — Peta spektrum seluler (20 menit)

**Teori:** Spektrum adalah sumber daya terbatas (slide 3). Operator mendapat blok frekuensi di pita tertentu. Pita FDD terdiri dari pasangan uplink (UL) dan downlink (DL).

```bash
hackrf_sweep -f 700:2700 -w 100000 -l 32 -g 30 -a 0 -N 30 -r data/sweep_seluler.csv
python3 peta_seluler.py data/sweep_seluler.csv --out gambar/p1
```

Buka `gambar/p1_peta.png` dan `gambar/p1_carrier.csv`.

**Amati dan catat**

| Pita | DL (MHz) | Carrier lebar yang terlihat (frekuensi tengah, lebar) | Tebakan bandwidth LTE |
|---|---|---|---|
| B8 900 | 925–960 | | |
| B3 1800 | 1805–1880 | | |
| B1 2100 | 2110–2170 | | |
| B40 2300 (TDD) | 2300–2400 | | |

**Pertanyaan**
1. Mengapa lebar terisi carrier LTE 20 MHz hanya ±18 MHz? (Petunjuk: 100 RB × 180 kHz.)
2. Garis sempit yang tepat di MHz bulat (misalnya 1993,0 dan 1997,0 MHz) muncul juga di luar pita seluler. Lepas antena lalu ulangi sweep: apakah garis itu tetap ada? Apa artinya?
3. Kolom `earfcn` dihitung dengan EARFCN = N_offs + 10·(f − F_low). Hitung secara manual EARFCN untuk carrier B3 di 1870 MHz (F_low = 1805, N_offs = 1200).

## Percobaan 2 — Downlink vs uplink (15 menit)

**Teori:** Pada FDD, eNodeB memancar terus di DL. Ponsel hanya memancar di UL saat dijadwalkan. Jarak duplex B3 = 95 MHz, B8 = 45 MHz, B1 = 190 MHz.

```bash
hackrf_sweep -f 1700:1890 -w 100000 -l 32 -g 30 -a 0 -N 300 -r data/sweep_b3.csv
python3 peta_seluler.py data/sweep_b3.csv --out gambar/p2_b3 --waterfall
```

Saat dosen menjalankan sweep (±1 menit), satu mahasiswa meletakkan ponselnya 30 cm dari antena dan **mengunggah video**, lalu menjauhkan ponselnya.

**Pertanyaan**
1. Bandingkan pola DL dan UL di `p2_b3_waterfall.png`. Mengapa UL tampak seperti titik-titik singkat?
2. Apakah ada burst UL yang muncul tepat saat ponsel mengunggah? Di frekuensi berapa? Periksa di aplikasi info sel apakah ponsel memang memakai B3.
3. Garis vertikal tipis yang konstan di pita UL itu sinyal atau spur? Bagaimana membuktikannya?

## Percobaan 3 — Bentuk sinyal LTE: OFDMA (25 menit)

**Teori:** OFDMA membagi kanal menjadi subcarrier 15 kHz (slide 15–19). Setiap simbol OFDM berdurasi T_u = 1/15 kHz = 66,7 µs, ditambah *cyclic prefix* (CP), yaitu salinan ekor simbol yang ditempel di depan. Tujuh simbol membentuk satu slot 0,5 ms.

Pilih satu carrier lebar dari Percobaan 1 dan rekam **tepat di frekuensi tengahnya**:

```bash
hackrf_transfer -r data/lte1870.cs8 -f 1870000000 -s 20000000 -b 20000000 -n 20000000 -l 32 -g 24 -a 0
python3 bentuk_sinyal.py data/lte1870.cs8 --fc 1870 --out gambar/p3_lte1870
```

Untuk carrier 10 MHz, sample rate 15,36 Msps sudah cukup:
`hackrf_transfer -r data/lte945.cs8 -f 945200000 -s 15360000 -b 15000000 -n 7680000 -l 16 -g 16 -a 0`, lalu `python3 bentuk_sinyal.py data/lte945.cs8 --fs 15.36e6 --fc 945.2 --out gambar/p3_lte945`.

**Amati**
- `_spektrum.png`: blok rata bertepi tajam. Catat lebar terisi dan jumlah RB.
- `_spektrogram.png`: garis putih tiap 1 ms = subframe. Cari (a) garis vertikal tipis berulang, yaitu *reference signal* (CRS) yang selalu dipancarkan; (b) blok sempit ±0,5 MHz di tengah, yaitu PSS/SSS/PBCH; (c) blok-blok lebar = data pengguna (PDSCH) yang dijadwalkan per subframe.
- `_ofdm.png` kiri: puncak autokorelasi pada lag T_u. Kanan: tujuh puncak per 0,5 ms.

**Pertanyaan**
1. Dari T_u terukur, hitung spasi subcarrier Δf = 1/T_u. Bandingkan dengan 15 kHz di slide 17.
2. Jarak antar-puncak di grafik kanan ±71,4 µs, bukan 66,7 µs. Dari mana selisih 4,7 µs itu?
3. Sel yang sepi (contoh 945 MHz) menunjukkan puncak autokorelasi tambahan di 22 dan 44 µs. Petunjuk: CRS hanya mengisi setiap subcarrier ke-3. Jelaskan mengapa itu menghasilkan puncak di T_u/3.
4. Mengapa OFDMA cocok untuk downlink tetapi uplink LTE memakai SC-FDMA? (Petunjuk: lihat fluktuasi daya di `_selubung.png`, lalu kaitkan dengan baterai ponsel.)

## Percobaan 4 — Cell search: menemukan PCI (20 menit)

**Teori:** Ponsel menemukan sel dengan mencari PSS (3 kemungkinan, N_ID_2) lalu SSS (168 kemungkinan, N_ID_1). PCI = 3·N_ID_1 + N_ID_2. LTE memakai *frequency reuse 1*: sel-sel bertetangga berada di frekuensi yang sama dan dibedakan oleh PCI. Bandingkan dengan GSM yang memakai frekuensi berbeda per sel (slide 4–5).

Setelah itu ponsel membaca **MIB** (Master Information Block) dari PBCH di 6 RB tengah subframe 0. MIB berisi bandwidth sel, konfigurasi PHICH, dan nomor frame (SFN). Jumlah port antena tidak tertulis di MIB, tetapi tersimpan sebagai *mask* pada CRC-nya. Skrip hanya menerima MIB yang CRC-nya cocok, jadi MIB yang terbaca sekaligus membuktikan bahwa PCI dan timing frame benar. Dekoder PBCH ada di `../lte_cell_search/lte_common.py`; folder itu harus ikut disalin.

```bash
python3 cari_sel_lte.py data/lte1870.cs8 --fc 1870 --out gambar/p4_lte1870
```

**Catat**

| PCI | N_ID_1 | N_ID_2 | FDD/TDD | CFO (kHz) | ppm | Bandwidth (MIB) | Port antena | SFN pertama |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

**Pertanyaan**
1. Mengapa PSS muncul tiap 5 ms di `_pss.png`, padahal frame LTE 10 ms?
2. Berapa sel yang terdengar di satu frekuensi? Apa artinya bagi konsep sel di slide 4–5?
3. Buka aplikasi info sel di ponsel. Apakah PCI dan EARFCN sel layanan cocok dengan hasil SDR?
4. CFO dalam ppm adalah kesalahan osilator HackRF. Apakah nilainya sama untuk carrier di pita lain (945 MHz, 1870 MHz)? Mengapa harus sama?
5. Bandingkan bandwidth dari MIB dengan lebar terisi yang kamu ukur di Percobaan 3. Cocokkah dengan N_RB × 180 kHz?
6. SFN pada frame berturut-turut naik berapa? Berapa lama sampai SFN kembali ke 0? (SFN 10 bit.)
7. MIB menyebut 2 port antena. Hubungkan dengan MIMO (slide 20–21): mengapa satu HackRF dengan satu antena tetap bisa membaca PBCH yang dipancarkan dari 2 antena? (Petunjuk: SFBC.)
8. Bila SSS meragukan, skrip mencoba ke-168 N_ID_1 dan memeriksa CRC MIB. Mengapa CRC 16 bit cukup untuk mencegah PCI yang salah ikut terlapor?

## Percobaan 5 — FDD vs TDD (15 menit)

**Teori:** B40 (2300 MHz) memakai TDD: DL dan UL berbagi frekuensi yang sama secara bergantian waktu.

```bash
hackrf_transfer -r data/tdd2330.cs8 -f 2330000000 -s 20000000 -b 20000000 -n 10000000 -l 40 -g 30 -a 0
python3 bentuk_sinyal.py data/tdd2330.cs8 --fc 2330 --out gambar/p5_tdd2330
```

**Pertanyaan**
1. Dari `p5_tdd2330_selubung.png`: berapa periode pola on/off dan berapa persen waktu untuk DL?
2. Konfigurasi TDD LTE 0–6 (TS 36.211 Tabel 4.2-2) menentukan subframe D/S/U. Konfigurasi mana yang cocok dengan pengukuranmu?
3. Kadang terlihat daya kecil di dalam jeda. Sinyal apakah itu?
4. Bandingkan dengan `p3_lte1870_selubung.png` (FDD).

## Percobaan 6 — GSM: TDMA dan 200 kHz (20 menit, opsional)

**Teori:** GSM membagi pita menjadi carrier 200 kHz. Setiap carrier dibagi menjadi frame TDMA 4,615 ms yang terdiri dari 8 timeslot × 577 µs. Carrier BCCH memancarkan FCCH, yaitu nada murni +67,7 kHz, pada frame 0, 10, 20, 30, 40 dari multiframe 51.

Cari carrier sempit di pita B8 DL (Percobaan 1), lalu rekam di sekitarnya:

```bash
hackrf_transfer -r data/gsm951.cs8 -f 951000000 -s 4000000 -b 3500000 -n 4000000 -l 16 -g 16 -a 0
python3 gsm_fcch.py data/gsm951.cs8 --fc 951 --out gambar/p6_gsm
```

**Pertanyaan**
1. Dari selang FCCH, hitung panjang frame TDMA dan timeslot. Bandingkan dengan teori.
2. Mengapa selangnya 10-10-10-10-11 frame, bukan 10 terus?
3. Bandingkan efisiensi: satu carrier LTE 20 MHz vs 100 carrier GSM 200 kHz. Apa keuntungan OFDMA?
4. CFO yang terukur dari nada FCCH: apakah sama (dalam ppm) dengan hasil Percobaan 4?

## Percobaan 7 — Siapa saja yang memancar di sekitar kita? (20 menit)

**Teori:** Percobaan 1–4 dirangkum dalam satu perintah. `survei_sel.py` melakukan sweep 700–2700 MHz, mengenali carrier LTE dari tepi blok yang curam, merekam 0,3 s di tiap carrier dengan gain otomatis, lalu mencari PCI dan membaca MIB. Hasilnya adalah daftar sel yang terdengar dari tempat kamu berdiri.

```bash
python3 survei_sel.py --out gambar/survei          # ±1 menit, HackRF harus terhubung
```

Keluaran: `gambar/survei_inventaris.csv` (satu baris per sel), `survei_inventaris.png` (peta per pita, warna sama = PCI sama), dan `survei_catatan.txt` (inferensi otomatis).

**Membaca hasil: sinyal ini dari mana?**
- **PCI sama di beberapa carrier** biasanya berarti satu sektor yang memancar di beberapa pita sekaligus.
- **Dua PCI di satu carrier** dibandingkan langsung dalam satu rekaman. Bila awal frame atau SFN-nya berbeda, keduanya berasal dari **eNodeB berbeda**.
- **Jam jaringan (fase SFN).** SFN naik tiap 10 ms dan berulang tiap 10,24 s. Fase = (waktu awal frame − 10 ms × SFN) mod 10 240 ms, dengan waktu dari jam laptop. Jaringan yang menyelaraskan SFN ke waktu GPS memberi fase ≈ 7600 ms. Hasil ukur biasanya ±30 ms lebih kecil karena jeda start HackRF; cek jam laptop dengan `sntp time.apple.com`. Jam berbeda membuktikan eNodeB berbeda. Jam sama **belum** membuktikan satu eNodeB, karena semua eNodeB yang tersinkron GPS jamnya sama.
- **PCI bukan alamat:** hanya 504 nilai dan dipakai ulang di seluruh kota. Identitas global (operator + eNB ID) ada di SIB1, dan itu tidak dibaca di praktikum ini.

**Langkah dengan ponsel** (dikerjakan mahasiswa dengan SIM operator yang berbeda-beda, memakai tabel survei yang ditampilkan dosen):
1. Buka NetMonster atau Network Cell Info. Catat EARFCN, PCI, operator, eNB ID, dan Timing Advance (TA) sel layanan dan sel tetangga.
2. Cocokkan EARFCN + PCI dengan `survei_inventaris.csv`, lalu isi kolom operator dan eNB ID.
3. Cari eNB ID di cellmapper.net atau OpenCelliD untuk melihat lokasi tower dan arah sektornya.
4. Gambar tower, sektor, dan gedung kampus pada satu peta.

| Pita | MHz | EARFCN | PCI | Bandwidth | Port | Jam | Operator (ponsel) | eNB ID (ponsel) | Lokasi tower |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

**Pertanyaan**
1. Berapa carrier dan berapa sel LTE yang terdengar? Pita mana yang paling ramai?
2. PCI mana yang muncul di lebih dari satu carrier? Apa artinya bagi operator?
3. Di carrier yang berisi dua PCI, bagaimana kamu tahu keduanya berasal dari eNodeB yang berbeda?
4. Apakah ada jam yang bukan waktu GPS? Mengapa sebuah eNodeB bisa tidak tersinkron GPS?
5. Mengapa SDR tidak bisa mengukur jarak ke tower, sedangkan ponsel bisa (Timing Advance)?
6. Carrier mana yang tidak berisi sel LTE? Dari bentuk sinyalnya (Percobaan 3 dan 5), apa dugaanmu?

---

## Menjalankan semua percobaan per lokasi (dosen)

Satu perintah menjalankan P1–P7 dengan rekaman yang lebih panjang dan menyimpan hasilnya per lokasi. Dengan begitu, hasil dari dua tempat bisa dibandingkan, misalnya lab (A) dan kampus (B):

```bash
python3 jalankan_semua.py --lokasi A          # ±10 menit, ±250 MB
python3 jalankan_semua.py --lokasi B
python3 jalankan_semua.py --lokasi B --langkah 6     # ulangi satu percobaan saja
```

| Percobaan | Rekaman default | Opsi |
|---|---|---|
| P1 sweep 700–2700 MHz | 60 sapuan | `--sweep-n` |
| P7 survei | 1 s per carrier | `--survei-detik` |
| P2 uplink B3 | sweep 120 s (unggah video dari ponsel selama ini) | `--uplink-detik` |
| P3/P4 LTE 20 MHz dan 10 MHz | 2 s dan 1 s | `--lte-detik` |
| P5 TDD | 1 s | `--tdd-detik` |
| P6 GSM | 5 s (±21 multiframe) | `--gsm-detik` |

Carrier untuk P3–P6 dipilih otomatis dari sweep dan survei: LTE 20 MHz dan 10 MHz terkuat yang MIB-nya terbaca, carrier terlebar di B40, dan garis GSM terkuat. Gain juga dipilih otomatis untuk menghindari clipping. Hasil tersimpan di `lokasi/<nama>/`: `data/` (sweep dan IQ), `gambar/` (semua grafik), `log.txt` (keluaran lengkap), dan `ringkasan.txt` (angka utama untuk dibandingkan antar-lokasi). Untuk membagikan ke mahasiswa, buat paketnya dengan `python3 buat_paket_mahasiswa.py --lokasi A B --zip`. Hasilnya folder `../untuk_mahasiswa/` dan `untuk_mahasiswa.zip`, yang berisi materi HTML, skrip analisis (tanpa skrip yang butuh HackRF), data dan contoh hasil tiap lokasi, serta `MULAI_DI_SINI.md` dengan perintah analisis yang sudah disesuaikan dengan rekamannya.

---

## Laporan

Satu laporan per kelompok, maksimal 8 halaman:
1. Tabel hasil Percobaan 1–7 dari demonstrasi, atau dari analisis ulang contoh data di laptop sendiri.
2. Satu gambar dari setiap percobaan, dengan anotasi tangan yang menunjuk bagian yang dibahas.
3. Jawaban semua pertanyaan.
4. Peta kampus dari Percobaan 7: tower, sektor, dan operator.
5. Diskusi: bagian mana dari arsitektur EPS (eNodeB, MME, SGW, PGW, slide 11–14) yang **bisa** dan **tidak bisa** diamati dengan SDR? Mengapa?

## Kalau ada masalah

| Gejala | Penyebab | Solusi |
|---|---|---|
| `hackrf_open() failed` | Perangkat dipakai program lain / kabel | Tutup program lain, cabut-colok USB |
| `PERINGATAN ... clipping` | Gain terlalu tinggi | Turunkan `-l` 8 dB atau `-g` 6 dB |
| Spektrum datar, tidak ada carrier | Gain terlalu rendah / antena lepas | Naikkan gain, periksa konektor SMA |
| `Tidak ada PSS di atas ambang` | Rekaman tidak tepat di tengah carrier | Gunakan frekuensi tengah dari `p1_carrier.csv` |
| PCI ditulis `?` | SSS kurang jelas dan tidak ada PCI yang lolos CRC MIB | Persempit pencarian CFO dengan `--ppm <nilai dari P4>`: pencarian ±50 kHz bisa terkunci di CFO yang salah karena sifat Zadoff-Chu. Di lokasi uji, carrier TDD 2310/2330 MHz baru terbaca setelah ini (LTE TDD, PCI 278). Bila tetap `?`, rekam lebih lama atau perbaiki posisi antena |
| `MIB gagal` | Sinyal lemah atau timing frame meleset | Naikkan gain (tanpa clipping), rekam lebih lama |
| `lte_common.py tidak ditemukan` | Folder `lte_cell_search/` tidak ikut disalin | Salin folder itu ke sebelah `praktikum_seluler/` |
| Lonjakan tepat di tengah spektrum | DC offset HackRF | Normal; bukan sinyal |

## Isi folder

| Berkas | Fungsi |
|---|---|
| `seluler_dasar.py` | Tabel pita Indonesia, EARFCN, pembaca `.cs8`/sweep, PSS/SSS, gaya grafik |
| `peta_seluler.py` | Percobaan 1–2 |
| `bentuk_sinyal.py` | Percobaan 3 dan 5 |
| `cari_sel_lte.py` | Percobaan 4 (memakai `../lte_cell_search/lte_common.py` untuk MIB) |
| `gsm_fcch.py` | Percobaan 6 |
| `survei_sel.py` | Percobaan 7: inventaris sel (memakai `cari_sel_lte.py` dan `peta_seluler.py`) |
| `jalankan_semua.py` | Semua percobaan sekaligus, rekaman panjang, hasil di `lokasi/<nama>/` |
| `buat_paket_mahasiswa.py` | Siapkan folder `../untuk_mahasiswa/` (+ zip) berisi materi, skrip analisis, data lokasi, dan `MULAI_DI_SINI.md` |
| `slide/praktikum_seluler.html` | Teori Bab 5 (isi Slide SELULER.pptx) + praktikum P1–P7 dalam satu halaman HTML (buka di browser, bisa offline; navigasi di bilah atas). Bangun ulang dengan `python3 slide/bangun_slide.py` |
| `gambar/teori/` | Gambar dari Slide SELULER.pptx, dinamai sesuai nomor slide (`s05.png` = slide 5) |
| `data/` | Contoh rekaman dari pengukuran persiapan |
| `gambar/` | Contoh keluaran |
