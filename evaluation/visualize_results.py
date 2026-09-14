"""Visualization tools for qualitative segmentation results."""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_COLORS = {
    0: (220, 50, 50),
    1: (50, 180, 50),
    2: (50, 200, 220),
}

CLASS_NAMES = {
    0: "Necrosis",
    1: "Normal",
    2: "Steatosis"
}


def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask_map: np.ndarray,
    alpha: float = 0.45,
    draw_contours: bool = True
) -> np.ndarray:
    overlay = image_rgb.copy()

    for cls_id, color in CLASS_COLORS.items():
        binary_mask = (mask_map == cls_id).astype(np.uint8)
        if np.sum(binary_mask) == 0:
            continue

        colored_region = np.zeros_like(image_rgb)
        colored_region[:] = color

        idx = binary_mask > 0
        overlay[idx] = (alpha * colored_region[idx] + (1.0 - alpha) * overlay[idx]).astype(np.uint8)

        if draw_contours:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def generate_comparative_figure(
    img_rgb: np.ndarray,
    gt_mask: np.ndarray,
    preds_dict: dict,
    save_path: Path,
    title_suffix: str = ""
):
    num_panels = 2 + len(preds_dict)
    fig, axes = plt.subplots(1, num_panels, figsize=(4.2 * num_panels, 4.5), dpi=300)

    axes[0].imshow(img_rgb)
    axes[0].set_title("(a) Original Image", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    gt_overlay = overlay_mask_on_image(img_rgb, gt_mask)
    axes[1].imshow(gt_overlay)
    axes[1].set_title("(b) Ground Truth", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")

    letters = ["c", "d", "e", "f", "g"]
    for i, (model_name, p_mask) in enumerate(preds_dict.items()):
        ax = axes[2 + i]
        m_overlay = overlay_mask_on_image(img_rgb, p_mask)
        ax.imshow(m_overlay)
        panel_letter = letters[i] if i < len(letters) else str(i)
        ax.set_title(f"({panel_letter}) {model_name}", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=CLASS_NAMES[cid])
        for cid, color in CLASS_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figure saved to {save_path}")


def parse_yolo_label_to_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.full((height, width), 255, dtype=np.uint8)
    if not label_path.exists():
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        coords[:, 0] = coords[:, 0] * width
        coords[:, 1] = coords[:, 1] * height
        pts = coords.astype(np.int32)
        cv2.fillPoly(mask, [pts], cls_id)

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison figures")
    parser.add_argument("--img", type=str, help="Path to sample histology image patch")
    parser.add_argument("--gt", type=str, help="Path to ground truth label txt file")
    parser.add_argument("--out", type=str, default="results/figures/comparison_sample.png")
    args = parser.parse_args()

    if args.img and args.gt:
        img = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        gt = parse_yolo_label_to_mask(Path(args.gt), H, W)
        generate_comparative_figure(img, gt, {}, Path(args.out))
    else:
        print("[INFO] Run with --img and --gt to visualize specific test patches.")

