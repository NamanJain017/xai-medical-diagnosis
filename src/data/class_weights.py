"""
Class weight computation for NIH Chest X-ray14.

NIH Chest X-ray14 is severely imbalanced — Hernia appears in <0.2% of images
while Infiltration appears in ~18%. Without weighting, the model ignores rare
classes and achieves misleadingly high accuracy.

BCEWithLogitsLoss accepts a pos_weight tensor:
  pos_weight[i] = num_negatives[i] / num_positives[i]
This upweights the loss for positive examples of rare classes.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .dataset import CLASSES, CLASS_TO_IDX


def compute_class_weights(labels_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss from the training labels DataFrame.

    Args:
        labels_df: Training split of Data_Entry_2017.csv

    Returns:
        pos_weight: Tensor of shape (14,) — one weight per disease class
    """
    n_total = len(labels_df)
    pos_counts = np.zeros(len(CLASSES), dtype=np.float64)

    for label_str in labels_df["Finding Labels"]:
        if label_str == "No Finding":
            continue
        for disease in str(label_str).split("|"):
            disease = disease.strip()
            if disease in CLASS_TO_IDX:
                pos_counts[CLASS_TO_IDX[disease]] += 1

    neg_counts = n_total - pos_counts

    # Clip to avoid division by zero for classes with 0 positives in split
    pos_counts = np.clip(pos_counts, 1, None)
    pos_weight = neg_counts / pos_counts

    print("\n[class_weights] Disease distribution in training set:")
    print(f"  {'Class':<22} {'Positive':>10} {'Negative':>10} {'Weight':>10}")
    print(f"  {'-'*55}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<22} {int(pos_counts[i]):>10,} {int(neg_counts[i]):>10,} {pos_weight[i]:>10.2f}")

    return torch.tensor(pos_weight, dtype=torch.float32)


def save_class_weights(weights: torch.Tensor, path: str):
    """Save class weights as numpy array."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, weights.numpy())
    print(f"\n[class_weights] Saved to {path}")


def load_class_weights(path: str) -> torch.Tensor:
    """Load saved class weights."""
    arr = np.load(path)
    return torch.tensor(arr, dtype=torch.float32)
