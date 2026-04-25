"""
Metric 2: Explanation Fidelity (Deletion Test / Confidence Drop)

Research question: Are the highlighted pixels actually responsible for the model's prediction?

Method:
  For K% ∈ {10, 20, 30, 50}:
    1. Record original model confidence S₀ for a correctly classified image
    2. Identify the top-K% most important pixels from the heatmap
    3. Replace those pixels with the training set mean pixel value (masking)
    4. Re-run model on the masked image, record S₁
    5. Fidelity(K) = mean(S₀ - S₁) over all test images

  Large drop = highlighted pixels were genuinely important (high fidelity)
  Small drop = explanation was misleading (model still confident without them)

  Plot: Fidelity curve (K% on x-axis, confidence drop on y-axis)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


# ImageNet channel means (pre-normalized space) — used as masking value
# In normalized space: (0.485 - 0.485) / 0.229 ≈ 0.0  (mean maps to ~0)
MASK_VALUE = 0.0


def mask_top_k_pixels(
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    k_percent: float,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """
    Replace the top-K% most important pixels in an image with a neutral value.

    Args:
        img_tensor:  (C, H, W) normalized image tensor
        heatmap:     (H, W) explanation heatmap in [0, 1]
        k_percent:   Percentage of pixels to mask (e.g., 20.0)
        mask_value:  Value to replace masked pixels with

    Returns:
        masked_img: (C, H, W) tensor with top-K% pixels replaced
    """
    H, W = heatmap.shape
    n_pixels = H * W
    k = max(1, int(n_pixels * k_percent / 100.0))

    # Flatten heatmap and find top-K pixel indices
    flat_heatmap = heatmap.flatten()
    top_k_indices = np.argpartition(flat_heatmap, -k)[-k:]

    # Convert flat indices to 2D
    rows = top_k_indices // W
    cols = top_k_indices % W

    # Clone tensor and mask
    masked = img_tensor.clone()
    masked[:, rows, cols] = mask_value  # mask all channels at those pixel locations

    return masked


@torch.no_grad()
def compute_fidelity_single(
    model,
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    class_idx: int,
    k_values: List[float],
    device: torch.device,
) -> dict:
    """
    Compute fidelity scores for a single image at multiple K values.

    Args:
        model:      Trained model
        img_tensor: (C, H, W) normalized image tensor
        heatmap:    (H, W) explanation heatmap in [0, 1]
        class_idx:  Disease class index to measure confidence for
        k_values:   List of K% values to test (e.g., [10, 20, 30, 50])
        device:     torch.device

    Returns:
        dict: {k: confidence_drop} for each K value
    """
    model.eval()
    input_batch = img_tensor.unsqueeze(0).to(device)

    # Original confidence S₀
    logits = model(input_batch)
    s0 = torch.sigmoid(logits)[0, class_idx].item()

    fidelity = {}
    for k in k_values:
        masked_img  = mask_top_k_pixels(img_tensor, heatmap, k_percent=k)
        masked_batch = masked_img.unsqueeze(0).to(device)

        logits_masked = model(masked_batch)
        s1 = torch.sigmoid(logits_masked)[0, class_idx].item()

        fidelity[k] = max(0.0, s0 - s1)  # clamp at 0 (can't go negative meaningfully)

    return {"s0": s0, "fidelity_per_k": fidelity}


def aggregate_fidelity_scores(
    all_scores: List[dict],
    k_values: List[float],
) -> dict:
    """
    Aggregate fidelity scores across all images.

    Returns:
        dict: {k: {"mean": ..., "std": ..., "n": ...}} for each K value
    """
    result = {}
    for k in k_values:
        drops = [s["fidelity_per_k"][k] for s in all_scores if k in s["fidelity_per_k"]]
        result[k] = {
            "mean": float(np.mean(drops)) if drops else float("nan"),
            "std":  float(np.std(drops))  if drops else float("nan"),
            "n":    len(drops),
        }
    return result
