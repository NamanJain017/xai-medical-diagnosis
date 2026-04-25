"""
ChestXrayDataset — PyTorch Dataset for NIH Chest X-ray14.

Handles:
  - Multi-label classification (14 disease classes, pipe-separated in CSV)
  - Bounding box annotations (984 images with ground-truth localization)
  - Grayscale → 3-channel RGB conversion
  - On-the-fly image loading (memory efficient for 112k images)
"""

import os
import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ── Disease class ordering (must match config) ─────────────────────────
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-ray14.

    Args:
        image_dir:    Path to the directory containing all .png images.
        labels_df:    DataFrame slice (rows for this split) from Data_Entry_2017.csv.
        bbox_df:      Full BBox_List_2017.csv DataFrame (None for train/val).
        transform:    torchvision transforms to apply.
        return_bbox:  If True, returns bounding box dict alongside image + label.
    """

    def __init__(
        self,
        image_dir: str,
        labels_df: pd.DataFrame,
        bbox_df: Optional[pd.DataFrame] = None,
        transform=None,
        return_bbox: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.bbox_df   = bbox_df
        self.transform = transform
        self.return_bbox = return_bbox

        # Pre-build bbox lookup: image_id → list of {label, x, y, w, h}
        self._bbox_lookup: dict = {}
        if bbox_df is not None:
            for _, row in bbox_df.iterrows():
                img_id = row["Image Index"]
                entry = {
                    "label": row["Finding Label"],
                    "x": float(row["Bbox [x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h]"]),
                }
                self._bbox_lookup.setdefault(img_id, []).append(entry)

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]
        image_id: str = row["Image Index"]

        # ── Load image ───────────────────────────────────────────────────
        img_path = self.image_dir / image_id
        img = Image.open(img_path).convert("RGB")  # grayscale → 3-channel

        if self.transform:
            img = self.transform(img)

        # ── Build multi-label target vector ──────────────────────────────
        label_str: str = row["Finding Labels"]
        target = torch.zeros(len(CLASSES), dtype=torch.float32)

        if label_str != "No Finding":
            for disease in label_str.split("|"):
                disease = disease.strip()
                if disease in CLASS_TO_IDX:
                    target[CLASS_TO_IDX[disease]] = 1.0

        if not self.return_bbox:
            return img, target, image_id

        # ── Bounding box (test set only) ──────────────────────────────────
        bbox_list = self._bbox_lookup.get(image_id, [])
        return img, target, image_id, bbox_list

    @staticmethod
    def collate_fn(batch):
        """Custom collate to handle variable-length bbox lists."""
        if len(batch[0]) == 3:
            imgs, targets, ids = zip(*batch)
            return torch.stack(imgs), torch.stack(targets), list(ids)
        else:
            imgs, targets, ids, bboxes = zip(*batch)
            return torch.stack(imgs), torch.stack(targets), list(ids), list(bboxes)


def load_dataframes(labels_file: str, bbox_file: str):
    """
    Load and clean the NIH CSV files.

    Returns:
        labels_df: cleaned Data_Entry_2017 DataFrame
        bbox_df:   cleaned BBox_List_2017 DataFrame
    """
    labels_df = pd.read_csv(labels_file)
    # Normalize column names
    labels_df.columns = labels_df.columns.str.strip()

    bbox_df = pd.read_csv(bbox_file)
    bbox_df.columns = bbox_df.columns.str.strip()

    print(f"[dataset] Loaded {len(labels_df):,} total images")
    print(f"[dataset] Loaded {len(bbox_df):,} bounding box annotations "
          f"across {bbox_df['Image Index'].nunique()} images")

    return labels_df, bbox_df
