"""Tiling and annotation extraction for histopathology slides."""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from shapely.geometry import Polygon, box, MultiPolygon
from tqdm import tqdm

from stain_norm import MacenkoNormalizer


CLASS_MAP = {
    "necrosis": 0,
    "necrotic": 0,
    "nekrosis": 0,
    "normal": 1,
    "hepatocyte": 1,
    "normal_parenchyma": 1,
    "steatosis": 2,
    "fatty": 2,
    "perlemakan": 2,
    "lipid": 2
}


def is_background_patch(patch_rgb: np.ndarray, bg_threshold: float = 220, max_bg_ratio: float = 0.85) -> bool:
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    bg_mask = gray > bg_threshold
    bg_ratio = np.mean(bg_mask)
    return bg_ratio > max_bg_ratio


def parse_qupath_geojson(geojson_path: str):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    polygons = []
    features = data.get("features", []) if isinstance(data, dict) else data

    for feat in features:
        props = feat.get("properties", {})
        classification = props.get("classification", {})
        class_name = classification.get("name", "").lower() if isinstance(classification, dict) else str(classification).lower()

        matched_class = None
        for key, val in CLASS_MAP.items():
            if key in class_name:
                matched_class = val
                break

        if matched_class is None:
            continue

        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            exterior = coords[0]
            if len(exterior) >= 3:
                poly = Polygon(exterior)
                if poly.is_valid and poly.area > 10:
                    polygons.append((poly, matched_class))
        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                exterior = poly_coords[0]
                if len(exterior) >= 3:
                    poly = Polygon(exterior)
                    if poly.is_valid and poly.area > 10:
                        polygons.append((poly, matched_class))

    return polygons


def tile_image_and_annotations(
    img_path: Path,
    annotation_path: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    patch_size: int = 512,
    normalize_stain: bool = True,
    normalizer: MacenkoNormalizer = None
):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[WARN] Unable to read image: {img_path}")
        return 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    polygons_with_class = []
    if annotation_path and annotation_path.exists():
        polygons_with_class = parse_qupath_geojson(str(annotation_path))

    patch_count = 0
    base_name = img_path.stem

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch_rgb = img_rgb[y:y + patch_size, x:x + patch_size]

            if is_background_patch(patch_rgb):
                continue

            patch_box = box(x, y, x + patch_size, y + patch_size)
            patch_annotations = []

            for poly, cls_idx in polygons_with_class:
                if not poly.intersects(patch_box):
                    continue

                inter = poly.intersection(patch_box)
                if inter.is_empty or inter.area < 20:
                    continue

                inter_polys = [inter] if isinstance(inter, Polygon) else [p for p in inter.geoms if isinstance(p, Polygon)]

                for p in inter_polys:
                    coords = np.array(p.exterior.coords)
                    local_x = np.clip((coords[:, 0] - x) / patch_size, 0.0, 1.0)
                    local_y = np.clip((coords[:, 1] - y) / patch_size, 0.0, 1.0)

                    poly_norm = Polygon(np.column_stack((local_x, local_y))).simplify(0.002, preserve_topology=True)
                    if poly_norm.is_empty or len(poly_norm.exterior.coords) < 3:
                        continue

                    coords_simp = np.array(poly_norm.exterior.coords)[:-1]
                    flat_coords = []
                    for pt_x, pt_y in coords_simp:
                        flat_coords.extend([f"{pt_x:.6f}", f"{pt_y:.6f}"])

                    if len(flat_coords) >= 6:
                        patch_annotations.append(f"{cls_idx} " + " ".join(flat_coords))

            if normalize_stain and normalizer is not None:
                patch_rgb = normalizer.transform(patch_rgb)

            patch_filename = f"{base_name}_x{x}_y{y}.png"
            patch_img_out = output_img_dir / patch_filename
            cv2.imwrite(str(patch_img_out), cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))

            patch_lbl_out = output_lbl_dir / f"{base_name}_x{x}_y{y}.txt"
            with open(patch_lbl_out, 'w', encoding='utf-8') as f:
                f.write("\n".join(patch_annotations))

            patch_count += 1

    return patch_count


def process_dataset(
    raw_images_dir: str,
    raw_annotations_dir: str,
    output_dir: str,
    patch_size: int = 512,
    normalize_stain: bool = True
):
    raw_img_path = Path(raw_images_dir)
    raw_ann_path = Path(raw_annotations_dir) if raw_annotations_dir else None
    out_path = Path(output_dir)

    out_img_dir = out_path / "images"
    out_lbl_dir = out_path / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    normalizer = MacenkoNormalizer() if normalize_stain else None

    image_files = list(raw_img_path.glob("*.png")) + list(raw_img_path.glob("*.jpg")) + list(raw_img_path.glob("*.tif*"))
    print(f"[INFO] Found {len(image_files)} raw images in {raw_img_path}")

    total_patches = 0
    for img_file in tqdm(image_files, desc="Tiling images"):
        ann_file = None
        if raw_ann_path:
            cand_ann = raw_ann_path / f"{img_file.stem}.geojson"
            if cand_ann.exists():
                ann_file = cand_ann

        count = tile_image_and_annotations(
            img_file,
            ann_file,
            out_img_dir,
            out_lbl_dir,
            patch_size=patch_size,
            normalize_stain=normalize_stain,
            normalizer=normalizer
        )
        total_patches += count

    print(f"[INFO] Generated {total_patches} patches in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile histopathology images into patches for YOLOv8-seg")
    parser.add_argument("--raw_images", type=str, default="data/liver_primary/raw_images", help="Path to raw slide images")
    parser.add_argument("--raw_annotations", type=str, default="data/liver_primary/raw_annotations", help="Path to QuPath GeoJSON annotations")
    parser.add_argument("--output_dir", type=str, default="data/liver_primary/processed", help="Output directory")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size")
    parser.add_argument("--no_stain_norm", action="store_true", help="Disable Macenko stain normalization")

    args = parser.parse_args()
    process_dataset(
        raw_images_dir=args.raw_images,
        raw_annotations_dir=args.raw_annotations,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        normalize_stain=not args.no_stain_norm
    )

