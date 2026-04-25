"""
Script 05: Compute all 3 evaluation metrics for all 4 model-method combinations.

Evaluates:
  - ResNet50   + Grad-CAM
  - ResNet50   + SHAP
  - EfficientNet + Grad-CAM
  - EfficientNet + SHAP

Metrics:
  1. IoU Score        — Localization accuracy vs bounding box ground truth
  2. Fidelity Score   — Confidence drop at K ∈ {10, 20, 30, 50}% pixel deletion
  3. Stability Score  — Cosine similarity under Gaussian noise (5 samples, σ=0.02)

Output files:
  outputs/metrics/iou_scores.csv
  outputs/metrics/fidelity_scores.csv
  outputs/metrics/stability_scores.csv
  outputs/metrics/summary_table.csv     ← Main 4×3 result table

Usage:
  python scripts/05_compute_metrics.py
  python scripts/05_compute_metrics.py --smoke-test
  python scripts/05_compute_metrics.py --metric iou   # compute only IoU
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import ChestXrayDataset, CLASSES, CLASS_TO_IDX
from src.data.transforms import get_val_transforms
from src.models.resnet50 import build_resnet50
from src.models.efficientnet import build_efficientnet
from src.evaluation.iou_metric import (
    compute_iou_for_image, heatmap_to_mask, aggregate_iou_scores
)
from src.evaluation.fidelity_metric import compute_fidelity_single, aggregate_fidelity_scores
from src.evaluation.stability_metric import compute_stability_single, aggregate_stability_scores
from src.xai.gradcam import GradCAMExplainer
from src.xai.gradshap import GradientSHAPExplainer
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.gpu_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Compute XAI evaluation metrics")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--metric", choices=["iou", "fidelity", "stability", "all"],
                   default="all")
    return p.parse_args()


COMBINATIONS = [
    ("resnet50", "gradcam"),
    ("resnet50", "shap"),
    ("efficientnet", "gradcam"),
    ("efficientnet", "shap"),
]


def load_model(model_name, cfg, device):
    models_dir = Path(cfg["output"]["models_dir"])
    ckpt_path  = models_dir / model_name / "best.pth"
    if model_name == "resnet50":
        model = build_resnet50(num_classes=cfg["num_classes"], pretrained=False)
    else:
        model = build_efficientnet(num_classes=cfg["num_classes"], pretrained=False)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✓ Loaded {model_name} checkpoint")
    else:
        print(f"  ⚠ No checkpoint for {model_name} — using random weights (smoke test only)")
    return model.to(device)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device()

    eval_cfg  = cfg["evaluation"]
    k_values  = eval_cfg["fidelity_k_values"]
    iou_perc  = eval_cfg["iou_threshold_percentile"]
    stab_std  = eval_cfg["stability_noise_std"]
    stab_n    = eval_cfg["stability_n_samples"]

    splits_dir   = Path(cfg["data"]["splits_dir"])
    heatmaps_dir = Path(cfg["output"]["heatmaps_dir"])
    metrics_dir  = Path(cfg["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  XAI Evaluation — Computing All Metrics")
    if args.smoke_test:
        print("  ⚡ SMOKE TEST MODE")
    print("=" * 60)

    # ── Load test set ─────────────────────────────────────────────────
    test_df = pd.read_csv(splits_dir / "test.csv")
    bbox_df = pd.read_csv(splits_dir / "bbox_annotations.csv")

    # Build lookup: image_id → list of bbox dicts
    bbox_lookup = {}
    for _, row in bbox_df.iterrows():
        img_id = row["Image Index"]
        entry  = {"label": row["Finding Label"],
                   "x": float(row["Bbox [x"]), "y": float(row["y"]),
                   "w": float(row["w"]), "h": float(row["h]"])}
        bbox_lookup.setdefault(img_id, []).append(entry)

    bbox_ids = list(bbox_df["Image Index"].unique())
    if args.smoke_test:
        bbox_ids = bbox_ids[:cfg["smoke_test"]["n_heatmap_images"]]

    test_ds = ChestXrayDataset(
        Path(cfg["data"]["images_dir"]), test_df, bbox_df=bbox_df,
        transform=get_val_transforms(cfg["data"]["image_size"]),
        return_bbox=True,
    )
    img_to_idx = {row["Image Index"]: i for i, row in test_ds.labels_df.iterrows()}

    # ── Results containers ────────────────────────────────────────────
    iou_results      = {}  # combo → list of scores
    fidelity_results = {}  # combo → list of per-image dicts
    stability_results= {}  # combo → list of per-image dicts

    # ── Process each combination ──────────────────────────────────────
    for model_name, method in COMBINATIONS:
        combo = f"{model_name}_{method}"
        heatmap_dir = heatmaps_dir / combo

        print(f"\n── {combo.upper()} ──")

        if not heatmap_dir.exists() or len(list(heatmap_dir.glob("*.npy"))) == 0:
            print(f"  ⚠ No heatmaps found in {heatmap_dir}")
            print(f"  Run: python scripts/04_generate_heatmaps.py --model {model_name} --method {method}")
            continue

        # Load model for fidelity and stability metrics
        model = load_model(model_name, cfg, device)
        model.eval()

        # Build explainer for stability
        if method == "gradcam":
            target_layer = model.get_gradcam_target_layer()
            explainer    = GradCAMExplainer(model, target_layer, device)
            explain_fn   = lambda img: explainer.explain(img)
        else:
            xai_cfg = cfg["xai"]["gradshap"]
            explainer = GradientSHAPExplainer(model, device,
                                               n_baselines=xai_cfg["n_baselines"],
                                               noise_std=xai_cfg["noise_std"])
            explain_fn = lambda img: explainer.explain(img)

        iou_scores_combo      = []
        fidelity_scores_combo = []
        stability_scores_combo= []

        for image_id in tqdm(bbox_ids, desc=f"  {combo}", unit="img", ncols=90):
            heatmap_path = heatmap_dir / f"{Path(image_id).stem}.npy"
            if not heatmap_path.exists():
                continue

            heatmap = np.load(str(heatmap_path))
            bbox_list = bbox_lookup.get(image_id, [])

            if image_id not in img_to_idx:
                continue

            img_tensor, target, *_ = test_ds[img_to_idx[image_id]]

            # Get predicted class
            with torch.no_grad():
                logits = model(img_tensor.unsqueeze(0).to(device))
                probs  = torch.sigmoid(logits).squeeze()
            class_idx = int(probs.argmax().item())
            class_name = CLASSES[class_idx]

            # ── Metric 1: IoU ────────────────────────────────────────
            if args.metric in ("iou", "all"):
                iou = compute_iou_for_image(
                    heatmap, bbox_list, disease_label=class_name,
                    threshold_percentile=iou_perc,
                )
                iou_scores_combo.append(iou)

            # ── Metric 2: Fidelity ───────────────────────────────────
            if args.metric in ("fidelity", "all"):
                fid = compute_fidelity_single(
                    model, img_tensor, heatmap, class_idx, k_values, device
                )
                fidelity_scores_combo.append(fid)

            # ── Metric 3: Stability ──────────────────────────────────
            if args.metric in ("stability", "all"):
                stab = compute_stability_single(
                    img_tensor, explain_fn, heatmap,
                    n_samples=stab_n, noise_std=stab_std,
                )
                stability_scores_combo.append(stab)

        iou_results[combo]       = iou_scores_combo
        fidelity_results[combo]  = fidelity_scores_combo
        stability_results[combo] = stability_scores_combo

    # ── Save CSVs and print summary table ─────────────────────────────
    summary_rows = []

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Combination':<28} {'IoU':>8} {'Fid@20%':>9} {'Stability':>10}")
    print(f"  {'-'*58}")

    for model_name, method in COMBINATIONS:
        combo = f"{model_name}_{method}"

        iou_agg  = aggregate_iou_scores(iou_results.get(combo, []))
        fid_agg  = aggregate_fidelity_scores(fidelity_results.get(combo, []), k_values)
        stab_agg = aggregate_stability_scores(stability_results.get(combo, []))

        fid20 = fid_agg.get(20, {}).get("mean", float("nan"))

        print(f"  {combo:<28} "
              f"{iou_agg['mean']:>8.4f} "
              f"{fid20:>9.4f} "
              f"{stab_agg['mean']:>10.4f}")

        row = {
            "combination": combo, "model": model_name, "method": method,
            "iou_mean": iou_agg["mean"], "iou_std": iou_agg["std"],
            "fidelity_10_mean": fid_agg.get(10, {}).get("mean", np.nan),
            "fidelity_20_mean": fid_agg.get(20, {}).get("mean", np.nan),
            "fidelity_30_mean": fid_agg.get(30, {}).get("mean", np.nan),
            "fidelity_50_mean": fid_agg.get(50, {}).get("mean", np.nan),
            "stability_mean": stab_agg["mean"], "stability_std": stab_agg["std"],
            "n_images": iou_agg["n"],
        }
        summary_rows.append(row)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metrics_dir / "summary_table.csv", index=False)
    print(f"\n  ✅ Summary table → {metrics_dir / 'summary_table.csv'}")
    print(f"   Next: python scripts/06_analyze_results.py")


if __name__ == "__main__":
    main()
