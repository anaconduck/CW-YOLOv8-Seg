"""Patient/slide-level cross-validation."""

import os
import shutil
import argparse
from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import KFold

from finetune_liver import run_liver_finetuning


def setup_kfold_splits(processed_dir: Path, n_splits: int = 5, seed: int = 42):
    img_dir = processed_dir / "images"
    lbl_dir = processed_dir / "labels"

    patch_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    slide_ids = sorted(list(set([p.stem.split("_x")[0] for p in patch_files])))

    print(f"[INFO] Found {len(slide_ids)} unique slides generating {len(patch_files)} patches.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(slide_ids)):
        train_slides = set([slide_ids[i] for i in train_idx])
        val_slides = set([slide_ids[i] for i in val_idx])
        splits.append((fold_idx, train_slides, val_slides))

    return splits, patch_files


def run_kfold_experiments(
    processed_dir: str,
    pretrained_weights: str,
    n_splits: int = 5,
    epochs: int = 60,
    batch_size: int = 16,
    device: str = "0",
    loss_type: str = "wiou_v3"
):
    proc_path = Path(processed_dir)
    splits, patch_files = setup_kfold_splits(proc_path, n_splits=n_splits)

    fold_metrics = []

    for fold_idx, train_slides, val_slides in splits:
        print("\n" + "#" * 60)
        print(f"RUNNING FOLD {fold_idx + 1} / {n_splits}")
        print(f"Train Slides: {len(train_slides)} | Val Slides: {len(val_slides)}")
        print("#" * 60)

        fold_dir = proc_path.parent / f"kfold_splits" / f"fold_{fold_idx}"
        for split_type in ["train", "val"]:
            (fold_dir / "images" / split_type).mkdir(parents=True, exist_ok=True)
            (fold_dir / "labels" / split_type).mkdir(parents=True, exist_ok=True)

        for p_file in patch_files:
            slide_id = p_file.stem.split("_x")[0]
            split_type = "train" if slide_id in train_slides else "val"

            dst_img = fold_dir / "images" / split_type / p_file.name
            if not dst_img.exists():
                shutil.copy2(p_file, dst_img)

            lbl_file = proc_path / "labels" / f"{p_file.stem}.txt"
            if lbl_file.exists():
                dst_lbl = fold_dir / "labels" / split_type / f"{p_file.stem}.txt"
                if not dst_lbl.exists():
                    shutil.copy2(lbl_file, dst_lbl)

        fold_yaml = fold_dir / "data.yaml"
        yaml_content = {
            "path": str(fold_dir).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "necrosis", 1: "normal", 2: "steatosis"}
        }
        with open(fold_yaml, "w") as f:
            yaml.dump(yaml_content, f)

        result = run_liver_finetuning(
            pretrained_weights=pretrained_weights,
            data_yaml=str(fold_yaml),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            loss_type=loss_type,
            output_name=f"liver_cv_fold_{fold_idx}"
        )
        fold_metrics.append(result)

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross Validation")
    parser.add_argument("--processed_dir", type=str, default="data/liver_primary/processed")
    parser.add_argument("--weights", type=str, required=True, help="PanNuke pre-trained weights path")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--loss", type=str, default="wiou_v3")

    args = parser.parse_args()
    run_kfold_experiments(
        processed_dir=args.processed_dir,
        pretrained_weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        loss_type=args.loss
    )

