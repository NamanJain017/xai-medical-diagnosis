"""
Per-class AUC-ROC computation for multi-label chest X-ray classification.

NIH Chest X-ray14 is heavily imbalanced — overall accuracy is meaningless.
Per-class AUC-ROC is the standard metric in chest X-ray literature.

Target: mean AUC ≥ 0.80 across all 14 classes.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..data.dataset import CLASSES


def compute_auc(
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Compute per-class and mean AUC-ROC.

    Args:
        all_targets: (N, 14) binary ground truth
        all_probs:   (N, 14) sigmoid probabilities
        verbose:     print per-class breakdown

    Returns:
        dict with 'per_class' (dict) and 'mean' (float)
    """
    per_class_auc = {}
    valid_classes = []

    for i, cls in enumerate(CLASSES):
        y_true = all_targets[:, i]
        y_score = all_probs[:, i]

        # AUC undefined if only one class present in ground truth
        if len(np.unique(y_true)) < 2:
            per_class_auc[cls] = float("nan")
            continue

        auc = roc_auc_score(y_true, y_score)
        per_class_auc[cls] = auc
        valid_classes.append(auc)

    mean_auc = float(np.mean(valid_classes)) if valid_classes else 0.0

    if verbose:
        print(f"\n  {'Class':<24} {'AUC':>8}")
        print(f"  {'-'*35}")
        for cls, auc in per_class_auc.items():
            flag = " ✓" if not np.isnan(auc) and auc >= 0.80 else ""
            val = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
            print(f"  {cls:<24} {val:>8}{flag}")
        print(f"  {'-'*35}")
        print(f"  {'Mean AUC':<24} {mean_auc:>8.4f}")

    return {"per_class": per_class_auc, "mean": mean_auc}


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run inference over a DataLoader and collect all targets + probabilities.

    Returns:
        targets: (N, 14) numpy array
        probs:   (N, 14) numpy array  [sigmoid applied]
    """
    model.eval()
    all_targets, all_probs = [], []

    for batch in loader:
        imgs, targets, *_ = batch
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        probs  = torch.sigmoid(logits).cpu()

        all_targets.append(targets.numpy())
        all_probs.append(probs.numpy())

    return np.vstack(all_targets), np.vstack(all_probs)
