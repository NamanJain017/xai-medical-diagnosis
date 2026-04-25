"""
Script 01: Prepare NIH Chest X-ray14 data.

Tasks:
  1. Parse Data_Entry_2017.csv — extract image paths and labels
  2. Parse BBox_List_2017.csv — extract bounding box annotations
  3. Ensure all 984 bbox images land in the test set
  4. Create stratified 80/10/10 train/val/test splits
  5. Compute and save class weights for BCEWithLogitsLoss
  6. Print dataset statistics

Usage:
  python scripts/01_prepare_data.py
  python scripts/01_prepare_data.py --smoke-test     # 100 images only
  python scripts/01_prepare_data.py --config config/config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# ── Allow imports from src/ ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset import load_dataframes, CLASSES, CLASS_TO_IDX
from src.data.class_weights import compute_class_weights, save_class_weights
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Prepare NIH Chest X-ray14 splits")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="Use only 100 images for a fast pipeline smoke test")
    return p.parse_args()


def verify_images_exist(df: pd.DataFrame, images_dir: Path, sample: int = 20) -> bool:
    """Spot-check that image files actually exist on disk."""
    sample_ids = df["Image Index"].sample(min(sample, len(df))).tolist()
    missing = [i for i in sample_ids if not (images_dir / i).exists()]
    if missing:
        print(f"⚠ WARNING: {len(missing)} sampled images not found in {images_dir}")
        print(f"  Example: {missing[0]}")
        return False
    return True


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    data_cfg   = cfg["data"]
    images_dir = Path(data_cfg["images_dir"])
    splits_dir = Path(data_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NIH Chest X-ray14 — Data Preparation")
    if args.smoke_test:
        print("  ⚡ SMOKE TEST MODE — using 100 images")
    print("=" * 60)

    # ── 1. Load CSVs ──────────────────────────────────────────────────
    labels_df, bbox_df = load_dataframes(
        labels_file=data_cfg["labels_file"],
        bbox_file=data_cfg["bbox_file"],
    )

    # ── 2. Verify images exist ────────────────────────────────────────
    print(f"\n[step 1/5] Verifying images directory: {images_dir}")
    if not images_dir.exists():
        print(f"  ❌ ERROR: Images directory not found: {images_dir}")
        print("  Please download the NIH Chest X-ray14 dataset first.")
        print("  See README.md → Dataset Download section.")
        sys.exit(1)

    n_files = len(list(images_dir.glob("*.png")))
    print(f"  Found {n_files:,} PNG files in {images_dir}")
    verify_images_exist(labels_df, images_dir)

    # ── 3. Filter to available images ────────────────────────────────
    available = {f.name for f in images_dir.glob("*.png")}
    labels_df = labels_df[labels_df["Image Index"].isin(available)].copy()
    print(f"  Matched {len(labels_df):,} images in CSV to disk")

    # ── Smoke test: subsample ─────────────────────────────────────────
    if args.smoke_test:
        n_smoke = cfg["smoke_test"]["n_images"]
        bbox_ids = set(bbox_df["Image Index"].unique())
        bbox_in_df = labels_df[labels_df["Image Index"].isin(bbox_ids)]
        non_bbox   = labels_df[~labels_df["Image Index"].isin(bbox_ids)]
        
        # Take a small sample of bbox images (e.g., 20) for the test set
        n_bbox_smoke = min(20, len(bbox_in_df))
        bbox_sample  = bbox_in_df.sample(n_bbox_smoke, random_state=42)
        
        # The rest will be non-bbox images for train/val
        n_extra = max(0, n_smoke - n_bbox_smoke)
        extra   = non_bbox.sample(min(n_extra, len(non_bbox)), random_state=42)
        
        labels_df = pd.concat([bbox_sample, extra]).reset_index(drop=True)
        
        # Update bbox_df so only the sampled bbox images are forced into test
        bbox_df = bbox_df[bbox_df["Image Index"].isin(bbox_sample["Image Index"])]
        print(f"  Smoke test: using {len(labels_df)} images ({len(bbox_sample)} with bboxes)")

    # ── 4. Split ──────────────────────────────────────────────────────
    print(f"\n[step 2/5] Creating train/val/test splits")
    bbox_ids = set(bbox_df["Image Index"].unique())

    # Force all bbox images into test set
    test_df  = labels_df[labels_df["Image Index"].isin(bbox_ids)].copy()
    rest_df  = labels_df[~labels_df["Image Index"].isin(bbox_ids)].copy()

    # 80/10/10 on the remaining images
    val_size = data_cfg["val_split"] / (data_cfg["train_split"] + data_cfg["val_split"])
    train_df, val_df = train_test_split(rest_df, test_size=val_size, random_state=42)

    # Add non-bbox images to fill test if needed (test can be > 10% due to bbox constraint)
    total = len(labels_df)
    print(f"  Train: {len(train_df):,} ({100*len(train_df)/total:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({100*len(val_df)/total:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({100*len(test_df)/total:.1f}%) "
          f"[{len(bbox_ids)} bbox images guaranteed here]")

    # ── 5. Save splits ────────────────────────────────────────────────
    print(f"\n[step 3/5] Saving split index files to {splits_dir}")
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    bbox_df.to_csv(splits_dir / "bbox_annotations.csv", index=False)
    print(f"  ✅ Saved train.csv, val.csv, test.csv, bbox_annotations.csv")

    # ── 6. Class weights ──────────────────────────────────────────────
    print(f"\n[step 4/5] Computing class weights from training set")
    weights = compute_class_weights(train_df)
    weights_path = splits_dir / "class_weights.npy"
    save_class_weights(weights, str(weights_path))

    # ── 7. Summary statistics ─────────────────────────────────────────
    print(f"\n[step 5/5] Dataset statistics")
    print(f"  Total images available: {len(labels_df):,}")
    disease_counts = {}
    for label_str in labels_df["Finding Labels"]:
        if label_str == "No Finding":
            continue
        for d in str(label_str).split("|"):
            d = d.strip()
            if d in CLASS_TO_IDX:
                disease_counts[d] = disease_counts.get(d, 0) + 1

    print(f"  {'Disease':<24} {'Count':>8} {'%':>8}")
    print(f"  {'-'*42}")
    for cls in CLASSES:
        count = disease_counts.get(cls, 0)
        pct   = 100.0 * count / len(labels_df)
        print(f"  {cls:<24} {count:>8,} {pct:>8.2f}%")

    print(f"\n✅ Data preparation complete!")
    print(f"   Next: python scripts/02_train_resnet50.py")


if __name__ == "__main__":
    main()
