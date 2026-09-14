"""Pipeline runner for model training and evaluation."""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd_list, desc: str):
    print("\n" + "=" * 75)
    print(f"[STAGE] {desc}")
    print("Command: " + " ".join(cmd_list))
    print("=" * 75)
    ret = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        print(f"[ERROR] Stage '{desc}' failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="Pipeline runner for liver histopathology segmentation")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "download_pannuke",
            "tile_liver",
            "pretrain_pannuke",
            "finetune_liver",
            "kfold_cv",
            "ablation",
            "report"
        ],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Epoch count")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--weights", type=str, default=None, help="Pretrained weights path")

    args = parser.parse_args()
    py = sys.executable

    if args.stage == "download_pannuke":
        run_cmd([py, "data/download_pannuke.py"], "Download & Format PanNuke Dataset")

    elif args.stage == "tile_liver":
        run_cmd([py, "data/preprocessing/tiling.py", "--patch_size", "512"], "Tile Liver Slide Images into 512x512 Patches")

    elif args.stage == "pretrain_pannuke":
        run_cmd(
            [py, "training/pretrain_pannuke.py", "--epochs", str(args.epochs), "--batch", str(args.batch), "--device", args.device, "--loss", "wiou_v3"],
            "Pre-training on PanNuke"
        )

    elif args.stage == "finetune_liver":
        weights = args.weights or "results/pannuke_pretrained_carafe_wiou/weights/best.pt"
        run_cmd(
            [py, "training/finetune_liver.py", "--weights", weights, "--epochs", str(args.epochs), "--batch", str(args.batch), "--device", args.device, "--loss", "wiou_v3"],
            "Fine-tuning on Liver Dataset"
        )

    elif args.stage == "kfold_cv":
        weights = args.weights or "results/pannuke_pretrained_carafe_wiou/weights/best.pt"
        run_cmd(
            [py, "training/kfold_cv.py", "--weights", weights, "--epochs", str(args.epochs), "--batch", str(args.batch), "--device", args.device, "--loss", "wiou_v3"],
            "5-Fold Cross Validation"
        )

    elif args.stage == "ablation":
        print("[INFO] Running ablation study...")

    elif args.stage == "report":
        run_cmd([py, "evaluation/ablation_study.py"], "Generate Ablation Tables")


if __name__ == "__main__":
    main()

