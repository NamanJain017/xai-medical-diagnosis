"""
Metric 1: Localization Accuracy (IoU Score)

Research question: Does the explanation highlight the correct anatomical region?

Method:
  1. Threshold the heatmap at the 50th percentile → binary prediction mask
  2. Convert NIH bounding box annotation → binary ground truth mask
  3. IoU = Area(Intersection) / Area(Union)

Higher IoU = explanation is spatially aligned with actual disease location.
Only computable for the 984 bounding-box annotated images.
"""

import numpy as np
from typing import List, Dict


def bbox_to_mask(bbox: dict, image_size: int = 224) -> np.ndarray:
    """
    Convert a NIH bounding box annotation to a binary mask.

    The NIH BBox_List_2017.csv stores boxes in original image coordinates.
    We rescale to 224×224.

    Args:
        bbox: dict with keys 'x', 'y', 'w', 'h' (original image coords)
              and 'orig_w', 'orig_h' for the original image dimensions.
              If orig_w/orig_h missing, we assume 1024×1024 (NIH standard).
        image_size: output mask size

    Returns:
        mask: (image_size, image_size) binary numpy array
    """
    orig_w = bbox.get("orig_w", 1024)
    orig_h = bbox.get("orig_h", 1024)

    # Rescale coordinates to image_size
    scale_x = image_size / orig_w
    scale_y = image_size / orig_h

    x1 = int(bbox["x"] * scale_x)
    y1 = int(bbox["y"] * scale_y)
    x2 = int((bbox["x"] + bbox["w"]) * scale_x)
    y2 = int((bbox["y"] + bbox["h"]) * scale_y)

    # Clamp to image bounds
    x1, x2 = max(0, x1), min(image_size, x2)
    y1, y2 = max(0, y1), min(image_size, y2)

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def heatmap_to_mask(heatmap: np.ndarray, threshold_percentile: int = 50) -> np.ndarray:
    """
    Threshold a heatmap to produce a binary prediction mask.

    Args:
        heatmap:              (H, W) float32 array in [0, 1]
        threshold_percentile: Pixels above this percentile → 1, else 0

    Returns:
        mask: (H, W) binary numpy array
    """
    threshold = np.percentile(heatmap, threshold_percentile)
    return (heatmap >= threshold).astype(np.uint8)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    Args:
        pred_mask: (H, W) binary array — thresholded heatmap
        gt_mask:   (H, W) binary array — ground truth bounding box

    Returns:
        iou: float in [0, 1]. Returns 0.0 if both masks are empty.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union        = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_iou_for_image(
    heatmap: np.ndarray,
    bbox_list: List[dict],
    disease_label: str,
    image_size: int = 224,
    threshold_percentile: int = 50,
) -> float:
    """
    Compute IoU for a single image against the ground truth bbox for a specific disease.

    Args:
        heatmap:              (H, W) explanation heatmap in [0, 1]
        bbox_list:            List of bbox dicts for this image (from ChestXrayDataset)
        disease_label:        The disease class to evaluate (e.g., "Cardiomegaly")
        image_size:           Size of the image (224)
        threshold_percentile: Heatmap threshold percentile

    Returns:
        iou score, or NaN if no matching bbox found
    """
    # Find bboxes matching the disease label
    matching = [b for b in bbox_list if b["label"] == disease_label]
    if not matching:
        return float("nan")

    pred_mask = heatmap_to_mask(heatmap, threshold_percentile)

    # If multiple bboxes for same disease, use union of GT masks
    gt_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for bbox in matching:
        gt_mask = np.logical_or(gt_mask, bbox_to_mask(bbox, image_size)).astype(np.uint8)

    return compute_iou(pred_mask, gt_mask)


def aggregate_iou_scores(scores: List[float]) -> dict:
    """Compute mean and std of IoU scores, ignoring NaN."""
    valid = [s for s in scores if not np.isnan(s)]
    if not valid:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(valid)),
        "std":  float(np.std(valid)),
        "n":    len(valid),
    }
