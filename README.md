# CARAFE-Enhanced YOLOv8 with Wise-IoU Loss for Few-Shot Liver Histopathology Tissue Segmentation

Penelitian komputasi histopatologi hati (*Liver Histopathology*) berbasis Deep Learning untuk segmentasi jaringan klinis (**Necrosis**, **Normal**, **Steatosis**) dengan skenario data terbatas (~40 citra mikroskopis H&E tanpa augmentasi data).

---

## 🔬 Sorotan Metodologi (Novelty & Kontribusi)

1. **Dual Architectural Enhancement**:
   - **CARAFE Upsampling**: Menggantikan *bilinear upsampling* pada Feature Pyramid Network (FPN) YOLOv8 untuk rekonstruksi fitur yang peka terhadap konten (*content-aware*), menjaga ketajaman batas gradasi jaringan patologis.
   - **Wise-IoU (WIoU v3) Loss**: Menggantikan *CIoU loss* dengan mekanisme *dynamic non-monotonic focusing* untuk mendegradasi bobot sampel outlier/anotasi bergradasi kabur (*irregular boundaries*) yang khas pada histopatologi.
2. **Cross-Task Transfer Learning (PanNuke → Liver Tissue)**:
   - Tahap 1: Pre-training pada 189.744 anotasi nukleus dari 19 organ (*PanNuke dataset*).
   - Tahap 2: Fine-tuning pada data primer hati klinis rumah sakit (~40 gambar) dengan pembekuan *backbone* dan regularisasi kuat.
3. **Protokol Ketat Non-Augmentasi**:
   - Menjawab batasan "tanpa augmentasi" melalui teknik *non-overlapping tiling* (pemotongan sub-patch standar digital pathology 512×512) dan validasi silang 5-Fold pada level *slide* (*patient-level split*).
4. **Validasi Klinis 2 Patolog**:
   - Anotasi dilakukan oleh 2 patolog independen dengan evaluasi *Inter-Observer Agreement* (Cohen's Kappa & Dice Score).

---

## 📁 Struktur Direktori Proyek

```
Histopatologi/
├── data/
│   ├── liver.yaml                      # Konfigurasi dataset primer hati (3 kelas)
│   ├── pannuke.yaml                    # Konfigurasi dataset PanNuke (5 kelas)
│   ├── download_pannuke.py             # Downloader otomatis PanNuke dari Zenodo & converter YOLOv8
│   ├── liver_primary/
│   │   ├── raw_images/                 # Tempat meletakkan ~40 citra mikroskopis asli RS
│   │   ├── raw_annotations/            # Tempat file GeoJSON hasil ekspor QuPath
│   │   └── processed/                  # Hasil pemotongan tiling (images & labels YOLOv8-seg)
│   └── preprocessing/
│       ├── stain_norm.py               # Macenko Stain Normalization
│       ├── tiling.py                   # Pemotong patch 512x512 dan konversi polygon
│       └── qupath_export_script.groovy # Script otomatis 1-klik ekspor GeoJSON di QuPath
├── models/
│   ├── carafe_module.py                # Pure PyTorch CARAFE operator (CUDA accelerated)
│   ├── wiou_loss.py                    # Wise-IoU Loss (v1, v2, v3 dynamic focusing)
│   ├── yolov8_carafe_wiou.py           # Wrapper integrator Ultralytics YOLOv8
│   └── configs/
│       ├── yolov8s-seg-carafe.yaml     # Model utama (YOLOv8s + CARAFE di neck FPN)
│       └── yolov8n-seg-carafe.yaml     # Model nano (Ablation study)
├── training/
│   ├── pretrain_pannuke.py             # Stage 1: Pre-training pada PanNuke
│   ├── finetune_liver.py               # Stage 2: Fine-tuning data hati (Zero Augmentation)
│   └── kfold_cv.py                     # Runner 5-Fold Cross Validation (Slide-level split)
├── baselines/
│   └── unet_baseline.py                # External SOTA benchmark (U-Net) untuk pembanding Q1
├── evaluation/
│   ├── metrics.py                      # Evaluasi Dice, mIoU, Cohen's Kappa, Wilcoxon Test
│   ├── visualize_results.py            # Generator gambar visualisasi komparatif untuk paper
│   └── ablation_study.py               # Generator tabel ringkasan LaTeX (.tex) & Markdown
├── tests/
│   └── test_modules.py                 # Unit tests verifikasi tensor & gradien CARAFE & WIoU
├── results/                            # Checkpoints, log training, kurva metrik, & tabel
├── run_experiments.py                  # Master Pipeline Orchestrator CLI
├── requirements.txt                    # Dependensi Python
└── .gitignore                          # Filter file biner, dataset besar, dan .agents/
```

---

## 💻 Langkah 0: Persiapan Lingkungan & Cek GPU (RTX 5070)

Pastikan Python 3.10 atau 3.11 terpasang dan buka PowerShell di direktori `c:\Freelance\Histopatologi`:

```powershell
# 1. Masuk ke direktori proyek
cd c:\Freelance\Histopatologi

# 2. Buat dan aktifkan virtual environment (opsional namun disarankan)
python -m venv venv
.\venv\Scripts\activate

# 3. Pasang PyTorch dengan dukungan CUDA 12 untuk RTX 5070
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Pasang seluruh paket dependensi
pip install -r requirements.txt

# 5. Verifikasi bahwa GPU RTX 5070 terdeteksi oleh PyTorch
python -c "import torch; print('CUDA Tersedia:', torch.cuda.is_available()); print('Nama GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Tidak Ada GPU')"
```

---

## 🌐 Langkah 1: Download & Persiapan Dataset PanNuke (Pre-training)

Dataset PanNuke diperlukan untuk melatih representasi seluler histopatologi sebelum masuk ke data primer. Jalankan:

```bash
# Download dari Zenodo dan otomatis dikonversi ke format YOLOv8-seg:
python run_experiments.py --stage download_pannuke

# Opsi manual jika hanya ingin download zip saja:
python data/download_pannuke.py --download_only

# Opsi manual jika hanya ingin konversi npy yang sudah ada:
python data/download_pannuke.py --convert_only
```

Data terformat akan tersimpan otomatis di: `data/pannuke/yolo_format/`.

---

## 🩺 Langkah 2: Anotasi Data Primer oleh 2 Patolog (QuPath)

Data primer dari rumah sakit berjumlah ~40 citra mikroskopis H&E dengan delineasi 3 kelas klinis:

| Kelas | Kode | Kriteria Visual Mikroskopis H&E |
|---|---|---|
| **Necrosis** | 0 | Area kematian sel parenkim hati (hilangnya inti sel, sitoplasma hipereosinofilik/pudar, debris lisis seluler). |
| **Normal** | 1 | Lempeng hepatosit parenkim sehat teratur, batas membran sel tegas, sinusoid dan vena sentral utuh. |
| **Steatosis** | 2 | Degenerasi perlemakan hati (vakuola lipid bulat putih kosong/jernih intraseluler yang mendesak inti ke tepi). |

### Alur Kerja Anotasi:
1. Simpan ~40 gambar asli di: `data/liver_primary/raw_images/`
2. Buka software [QuPath](https://qupath.github.io/), buat **Project Baru** dan masukkan gambar.
3. Di panel sebelah kiri (**Annotations**), buat 3 kelas: `necrosis`, `normal`, `steatosis`.
4. Lakukan anotasi independen oleh **Patolog 1** dan **Patolog 2**.
5. Ekspor anotasi otomatis dengan 1-klik menggunakan script Groovy yang sudah disediakan:
   - Di QuPath, buka: **Automate** $\to$ **Show script editor**.
   - Buka file: [`data/preprocessing/qupath_export_script.groovy`](file:///c:/Freelance/Histopatologi/data/preprocessing/qupath_export_script.groovy).
   - Klik **Run** $\to$ **Run for project**.
   - Pindahkan file `.geojson` yang dihasilkan ke: `data/liver_primary/raw_annotations/`.

---

## 🧩 Langkah 3: Preprocessing Tiling & Normalisasi Warna

Potong citra besar menjadi sub-patch 512×512 dan selaraskan warna pewarnaan H&E:

```bash
python run_experiments.py --stage tile_liver
```
*Script ini otomatis menjalankan `data/preprocessing/tiling.py` dengan Macenko Stain Normalization, mengeliminasi background kosong, dan memformat poligon ke format YOLOv8-seg di `data/liver_primary/processed/`.*

---

## 🚀 Langkah 4: Stage 1 Training — Pre-training PanNuke

Latih model YOLOv8s-seg dengan CARAFE dan Wise-IoU Loss pada dataset PanNuke:

```bash
python run_experiments.py --stage pretrain_pannuke --epochs 100 --batch 16 --device 0
```
*Bobot pre-trained akan tersimpan di: `results/pannuke_pretrained_carafe_wiou/weights/best.pt`.*

---

## 🔬 Langkah 5: Stage 2 Training — Fine-tuning Data Hati (Zero Augmentation)

Lakukan transfer learning ke data primer hati dengan protokol ketat **tanpa augmentasi**:

```bash
python run_experiments.py --stage finetune_liver --epochs 80 --batch 16 --device 0
```
- **Kunci Non-Augmentasi**: Parameter `mosaic`, `mixup`, `fliplr`, `flipud`, `degrees`, dll dikunci pada nilai `0.0`.
- **Backbone Freezing**: Lapisan awal dibekukan untuk menjaga representasi fitur PanNuke dan mencegah overfitting.
- **Wise-IoU v3**: Mendegradasi bobot sampel outlier/batas tidak beraturan secara dinamis.

---

## 📊 Langkah 6: 5-Fold Cross-Validation & Uji Signifikansi Statistik

Jalankan validasi silang berbasis **Slide/Patient-level** (mencegah kebocoran data):

```bash
python run_experiments.py --stage kfold_cv --epochs 60 --batch 16 --device 0
```

Hitung metrik performa dan uji statistik untuk paper:
```bash
python evaluation/metrics.py
```
- **Dice Similarity Coefficient (DSC)** & **IoU** per kelas (`necrosis`, `normal`, `steatosis`).
- **Cohen's Kappa ($\kappa$)** antara Patolog 1 dan Patolog 2 (*Inter-Observer Agreement*).
- Nilai signifikansi statistik **Wilcoxon Signed-Rank Test** & **Paired t-test** ($p < 0.05$).

---

## 📄 Langkah 7: Pembuatan Gambar Publikasi & Tabel LaTeX Manuskrip

### 1. Generate Tabel LaTeX & Markdown Otomatis:
```bash
python run_experiments.py --stage report
```
File tabel akan dibuat secara otomatis di:
- `results/tables/ablation_summary.tex` (Tabel LaTeX siap salin langsung ke Overleaf)
- `results/tables/ablation_summary.md` (Tabel Markdown untuk preview cepat)

### 2. Generate Gambar Komparasi Visual Multi-Panel untuk Paper:
```bash
python evaluation/visualize_results.py --img data/liver_primary/processed/images/sample.png --gt data/liver_primary/processed/labels/sample.txt --out results/figures/figure_comparison.png
```
*Menghasilkan gambar beresolusi tinggi bersanding: (a) Original H&E Patch | (b) Ground Truth | (c) Baseline YOLOv8 | (d) YOLOv8 + CARAFE | (e) Proposed CARAFE + WIoU lengkap dengan legenda dan kontur.*
