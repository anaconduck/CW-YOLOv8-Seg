"""
Stage 1: Pre-training YOLOv8-CARAFE-WIoU on PanNuke Dataset.
Pre-trains backbone and neck features on 189,744 multi-tissue nuclei annotations.
Checkpoints will serve as weights initialization for Stage 2 liver fine-tuning.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.yolov8_carafe_wiou import register_carafe_to_ultralytics, patch_ultralytics_loss_with_wiou
from ultralytics import YOLO


def run_pretraining(
    data_yaml: str = "data/pannuke.yaml",
    model_cfg: str = "models/configs/yolov8s-seg-carafe.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 256,
    device: str = "0",
    loss_type: str = "wiou_v3",
    output_name: str = "pannuke_pretrained"
):
    print("=" * 70)
    print("STAGE 1: PanNuke Pre-training (Nuclei Instance Segmentation)")
    print(f"Model Configuration: {model_cfg}")
    print(f"Dataset: {data_yaml}")
    print(f"Device: {device} (RTX 5070)")
    print(f"BBox Loss: {loss_type}")
    print("=" * 70)

    # 1. Register CARAFE module in Ultralytics
    register_carafe_to_ultralytics()

    # 2. Patch loss with Wise-IoU if requested
    if "wiou" in loss_type.lower():
        v = 3 if "v3" in loss_type.lower() else (2 if "v2" in loss_type.lower() else 1)
        patch_ultralytics_loss_with_wiou(version=v)

    # 3. Initialize model
    model = YOLO(model_cfg)

    # 4. Run pre-training
    save_dir = PROJECT_ROOT / "results" / output_name
    save_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(PROJECT_ROOT / data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(PROJECT_ROOT / "results"),
        name=output_name,
        exist_ok=True,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        save=True,
        plots=True,
        seed=42,          # Fixed random seed for strict reproducibility
        deterministic=True,# Deterministic CUDA execution
        close_mosaic=10
    )

    print(f"[SUCCESS] Pre-training completed! Checkpoints saved at {save_dir / 'weights'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train YOLOv8-CARAFE on PanNuke dataset")
    parser.add_argument("--data", type=str, default="data/pannuke.yaml", help="Path to data yaml")
    parser.add_argument("--cfg", type=str, default="models/configs/yolov8s-seg-carafe.yaml", help="Path to model config")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (optimal for RTX 5070)")
    parser.add_argument("--imgsz", type=int, default=256, help="Image size (PanNuke standard is 256x256)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--loss", type=str, default="wiou_v3", choices=["wiou_v3", "wiou_v2", "wiou_v1", "ciou"])
    parser.add_argument("--name", type=str, default="pannuke_pretrained_carafe_wiou", help="Run name")

    args = parser.parse_args()
    run_pretraining(
        data_yaml=args.data,
        model_cfg=args.cfg,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        loss_type=args.loss,
        output_name=args.name
    )
