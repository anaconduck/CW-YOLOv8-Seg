"""Fine-tune model on liver histopathology dataset."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.yolov8_carafe_wiou import register_carafe_to_ultralytics, patch_ultralytics_loss_with_wiou
from ultralytics import YOLO


def run_liver_finetuning(
    pretrained_weights: str,
    data_yaml: str = "data/liver.yaml",
    epochs: int = 80,
    batch_size: int = 16,
    img_size: int = 512,
    device: str = "0",
    loss_type: str = "wiou_v3",
    freeze_backbone: bool = True,
    output_name: str = "liver_finetuned_carafe_wiou"
):
    print("=" * 75)
    print("Liver Histopathology Fine-Tuning")
    print(f"Weights: {pretrained_weights}")
    print(f"Dataset: {data_yaml}")
    print(f"Loss: {loss_type}")
    print(f"Freeze Backbone: {freeze_backbone}")
    print("=" * 75)

    register_carafe_to_ultralytics()

    if "wiou" in loss_type.lower():
        v = 3 if "v3" in loss_type.lower() else (2 if "v2" in loss_type.lower() else 1)
        patch_ultralytics_loss_with_wiou(version=v, alpha=1.9, delta=3.0)

    model = YOLO(pretrained_weights)
    freeze_arg = 10 if freeze_backbone else 0

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
        freeze=freeze_arg,
        workers=4,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.01,
        patience=20,
        save=True,
        plots=True,
        seed=42,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,
        crop_fraction=1.0
    )

    print(f"[INFO] Training finished. Weights saved in {save_dir / 'weights'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on liver histopathology dataset")
    parser.add_argument("--weights", type=str, required=True, help="Pre-trained weights (.pt)")
    parser.add_argument("--data", type=str, default="data/liver.yaml", help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=512, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--loss", type=str, default="wiou_v3", choices=["wiou_v3", "wiou_v2", "wiou_v1", "ciou"])
    parser.add_argument("--unfreeze", action="store_true", help="Unfreeze entire network")
    parser.add_argument("--name", type=str, default="liver_finetuned_carafe_wiou", help="Run name")

    args = parser.parse_args()
    run_liver_finetuning(
        pretrained_weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        loss_type=args.loss,
        freeze_backbone=not args.unfreeze,
        output_name=args.name
    )

