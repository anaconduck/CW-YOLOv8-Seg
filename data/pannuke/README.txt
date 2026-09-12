# Dataset PanNuke (Pre-training Stage 1)

Untuk mengunduh dan mengonversi PanNuke secara otomatis ke format YOLOv8-seg, jalankan:

    python data/download_pannuke.py

Opsi perintah:
- Unduh saja: python data/download_pannuke.py --download_only
- Konversi saja: python data/download_pannuke.py --convert_only

Output dataset akan tersimpan di folder:
    data/pannuke/yolo_format/
      ├── images/train (Fold 1 + Fold 2)
      ├── images/val   (Fold 3)
      ├── labels/train
      └── labels/val
