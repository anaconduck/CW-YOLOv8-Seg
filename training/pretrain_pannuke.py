"""Pre-training on PanNuke dataset."""

import sys
import argparse
from pathlib import Path

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
    print("PanNuke Pre-training")
    print(f"Model Configuration: {model_cfg}")
    print(f"Dataset: {data_yaml}")
    print(f"Loss: {loss_type}")
    print("=" * 70)

    register_carafe_to_ultralytics()

    if "wiou" in loss_type.lower():
        v = 3 if "v3" in loss_type.lower() else (2 if "v2" in loss_type.lower() else 1)
        patch_ultralytics_loss_with_wiou(version=v)

    model = YOLO(model_cfg)

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
        seed=42,
        deterministic=True,
        close_mosaic=10
    )

    print(f"[INFO] Pre-training finished. Weights saved in {save_dir / 'weights'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train model on PanNuke dataset")
    parser.add_argument("--data", type=str, default="data/pannuke.yaml", help="Path to data YAML")
    parser.add_argument("--cfg", type=str, default="models/configs/yolov8s-seg-carafe.yaml", help="Path to model config")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=256, help="Image size")
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

