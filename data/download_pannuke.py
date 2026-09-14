"""PanNuke dataset downloader and YOLOv8 segmentation converter."""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import urllib.request
import numpy as np
import cv2
from tqdm import tqdm

PANNUKE_URLS = {
    "fold1": "https://zenodo.org/records/3901844/files/Part%201.zip?download=1",
    "fold2": "https://zenodo.org/records/3901844/files/Part%202.zip?download=1",
    "fold3": "https://zenodo.org/records/3901844/files/Part%203.zip?download=1",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1000000:
        print(f"[INFO] {output_path.name} already exists. Skipping download.")
        return

    print(f"[INFO] Downloading {output_path.name} from {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"[INFO] Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def convert_fold_to_yolo(
    fold_dir: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    fold_prefix: str = "f1",
    min_polygon_points: int = 3
):
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_npy_candidates = list(fold_dir.rglob("images.npy"))
    mask_npy_candidates = list(fold_dir.rglob("masks.npy"))

    if not img_npy_candidates or not mask_npy_candidates:
        print(f"[WARN] Could not find images.npy or masks.npy inside {fold_dir}")
        return 0

    img_path = img_npy_candidates[0]
    mask_path = mask_npy_candidates[0]

    images = np.load(str(img_path))
    masks = np.load(str(mask_path))
    N, H, W, _ = images.shape

    converted_count = 0

    for i in tqdm(range(N), desc=f"Converting {fold_prefix}"):
        img_rgb = images[i].astype(np.uint8)
        mask_multi = masks[i]

        img_filename = f"{fold_prefix}_{i:05d}.png"
        cv2.imwrite(str(output_img_dir / img_filename), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        label_lines = []
        for cls_idx in range(5):
            cls_mask = mask_multi[:, :, cls_idx]
            unique_ids = np.unique(cls_mask)
            unique_ids = unique_ids[unique_ids > 0]

            for inst_id in unique_ids:
                inst_binary = (cls_mask == inst_id).astype(np.uint8)
                contours, _ = cv2.findContours(inst_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    epsilon = 0.015 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) >= min_polygon_points:
                        coords = approx.reshape(-1, 2)
                        norm_x = np.clip(coords[:, 0] / W, 0.0, 1.0)
                        norm_y = np.clip(coords[:, 1] / H, 0.0, 1.0)

                        flat_pts = []
                        for px, py in zip(norm_x, norm_y):
                            flat_pts.extend([f"{px:.5f}", f"{py:.5f}"])

                        if len(flat_pts) >= 6:
                            label_lines.append(f"{cls_idx} " + " ".join(flat_pts))

        lbl_filename = f"{fold_prefix}_{i:05d}.txt"
        with open(output_lbl_dir / lbl_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(label_lines))

        converted_count += 1

    return converted_count


def main():
    parser = argparse.ArgumentParser(description="Download and prepare PanNuke dataset for YOLOv8-seg")
    parser.add_argument("--data_dir", type=str, default="data/pannuke", help="Base directory to store PanNuke")
    parser.add_argument("--download_only", action="store_true", help="Only download zip files")
    parser.add_argument("--convert_only", action="store_true", help="Only convert existing .npy files")
    args = parser.parse_args()

    base_dir = Path(args.data_dir).resolve()
    zips_dir = base_dir / "zips"
    raw_dir = base_dir / "raw"
    yolo_dir = base_dir / "yolo_format"

    zips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.convert_only:
        for fold_name, url in PANNUKE_URLS.items():
            zip_dest = zips_dir / f"{fold_name}.zip"
            download_file(url, zip_dest)
            fold_extract = raw_dir / fold_name
            extract_zip(zip_dest, fold_extract)

    if args.download_only:
        return

    train_img_dir = yolo_dir / "images" / "train"
    train_lbl_dir = yolo_dir / "labels" / "train"

    f1_count = convert_fold_to_yolo(raw_dir / "fold1", train_img_dir, train_lbl_dir, fold_prefix="f1")
    f2_count = convert_fold_to_yolo(raw_dir / "fold2", train_img_dir, train_lbl_dir, fold_prefix="f2")

    val_img_dir = yolo_dir / "images" / "val"
    val_lbl_dir = yolo_dir / "labels" / "val"
    f3_count = convert_fold_to_yolo(raw_dir / "fold3", val_img_dir, val_lbl_dir, fold_prefix="f3")

    print(f"Dataset ready at: {yolo_dir}")
    print(f"Train samples: {f1_count + f2_count}, Val samples: {f3_count}")


if __name__ == "__main__":
    main()

