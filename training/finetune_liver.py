"""
Stage 2: Fine-tuning YOLOv8-CARAFE-WIoU on Primary Hospital Liver Histopathology Dataset.
Strictly adheres to NON-AUGMENTATION protocol (all augmentations disabled).
Applies transfer learning from Stage 1 PanNuke checkpoint with layer freezing and heavy regularization.
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
    print("STAGE 2: Liver Histopathology Tissue Fine-Tuning (Few-Shot / Small Dataset)")
    print(f"Pretrained Weights: {pretrained_weights}")
    print(f"Dataset: {data_yaml} (Classes: necrosis, normal, steatosis)")
    print(f"Protocol: STRICTLY NO DATA AUGMENTATION")
    print(f"Loss: {loss_type} (Wise-IoU Dynamic Non-Monotonic Focusing)")
    print(f"Freeze Backbone: {freeze_backbone}")
    print("=" * 75)

    # 1. Register CARAFE
    register_carafe_to_ultralytics()

    # 2. Patch Wise-IoU Loss
    if "wiou" in loss_type.lower():
        v = 3 if "v3" in loss_type.lower() else (2 if "v2" in loss_type.lower() else 1)
        patch_ultralytics_loss_with_wiou(version=v, alpha=1.9, delta=3.0)

    # 3. Load model with pre-trained weights
    model = YOLO(pretrained_weights)

    # 4. Freeze backbone layers if requested (layers 0 to 9 in YOLOv8)
    freeze_arg = 10 if freeze_backbone else 0

    save_dir = PROJECT_ROOT / "results" / output_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5. Train with ALL augmentations explicitly turned OFF (0.0)
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
        # Optimizer with weight decay regularization
        optimizer="AdamW",
        lr0=0.0001,       # Low learning rate for fine-tuning
        lrf=0.01,
        weight_decay=0.01,# High weight decay to combat overfitting on small data
        patience=20,      # Early stopping patience
        save=True,
        plots=True,
        seed=42,          # Locked random seed for strict reproducibility (Q1 standard)
        deterministic=True,# Enforce deterministic algorithms in PyTorch/cuDNN
        # ==========================================================
        # CRITICAL: STRICT NON-AUGMENTATION SETTINGS
        # ==========================================================
        mosaic=0.0,       # Disable mosaic
        mixup=0.0,        # Disable mixup
        copy_paste=0.0,   # Disable copy-paste
        degrees=0.0,      # Disable rotation
        translate=0.0,    # Disable translation
        scale=0.0,        # Disable scaling
        shear=0.0,        # Disable shearing
        perspective=0.0,  # Disable perspective transformation
        flipud=0.0,       # Disable vertical flip
        fliplr=0.0,       # Disable horizontal flip
        hsv_h=0.0,        # Disable color hue jitter
        hsv_s=0.0,        # Disable color saturation jitter
        hsv_v=0.0,        # Disable color brightness jitter
        erasing=0.0,      # Disable random erasing
        crop_fraction=1.0 # Use full patch without cropping
    )

    print(f"[SUCCESS] Fine-tuning completed! Final weights saved at {save_dir / 'weights'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8-CARAFE-WIoU on primary liver histopathology data")
    parser.add_argument("--weights", type=str, required=True, help="Path to PanNuke pre-trained weights (.pt)")
    parser.add_argument("--data", type=str, default="data/liver.yaml", help="Path to liver dataset YAML")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=512, help="Patch resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID (RTX 5070)")
    parser.add_argument("--loss", type=str, default="wiou_v3", choices=["wiou_v3", "wiou_v2", "wiou_v1", "ciou"])
    parser.add_argument("--unfreeze", action="store_true", help="Unfreeze entire network (full fine-tune)")
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
