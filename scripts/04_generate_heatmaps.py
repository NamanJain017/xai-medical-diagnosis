"""
Script 04: Generate and cache all 3,936 explanation heatmaps.

For each of the 984 bounding-box test images × 4 model-method combinations:
  1. ResNet50   + Grad-CAM   → outputs/heatmaps/resnet50_gradcam/{image_id}.npy
  2. ResNet50   + SHAP       → outputs/heatmaps/resnet50_shap/{image_id}.npy
  3. EfficientNet + Grad-CAM → outputs/heatmaps/efficientnet_gradcam/{image_id}.npy
  4. EfficientNet + SHAP     → outputs/heatmaps/efficientnet_shap/{image_id}.npy

Heatmaps are normalized to [0, 1] before saving.
SHAP is slow — precomputing and caching is critical.

Usage:
  python scripts/04_generate_heatmaps.py
  python scripts/04_generate_heatmaps.py --smoke-test     # 10 images only
  python scripts/04_generate_heatmaps.py --method gradcam # only Grad-CAM
  python scripts/04_generate_heatmaps.py --model resnet50 # only ResNet50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_val_transforms
from src.models.resnet50 import build_resnet50
from src.models.efficientnet import build_efficientnet
from src.xai.gradcam import GradCAMExplainer
from src.xai.gradshap import GradientSHAPExplainer
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.gpu_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Generate XAI heatmaps for all combinations")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="Process only 10 images")
    p.add_argument("--method", choices=["gradcam", "shap", "both"], default="both")
    p.add_argument("--model", choices=["resnet50", "efficientnet", "both"], default="both")
    return p.parse_args()


def load_model(model_name: str, cfg: dict, device: torch.device):
    """Load best checkpoint for a given model."""
    models_dir = Path(cfg["output"]["models_dir"])
    ckpt_path  = models_dir / model_name / "best.pth"

    if model_name == "resnet50":
        model = build_resnet50(num_classes=cfg["num_classes"], pretrained=False)
    else:
        model = build_efficientnet(num_classes=cfg["num_classes"], pretrained=False)

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"  Train the model first: python scripts/02_train_{model_name}.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Loaded {model_name} (epoch {ckpt['epoch']}, AUC={ckpt['val_auc']:.4f})")
    return model


def generate_for_combination(
    model,
    model_name: str,
    method: str,
    dataset: ChestXrayDataset,
    image_ids: list,
    cfg: dict,
    device: torch.device,
    smoke_n: int = None,
):
    """Generate and save heatmaps for one model-method combination."""
    combo_name = f"{model_name}_{method}"
    out_dir    = Path(cfg["output"]["heatmaps_dir"]) / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    xai_cfg = cfg["xai"]

    # Build explainer
    if method == "gradcam":
        target_layer = model.get_gradcam_target_layer()
        explainer    = GradCAMExplainer(model, target_layer, device)
    else:  # shap
        explainer = GradientSHAPExplainer(
            model, device,
            n_baselines=xai_cfg["gradshap"]["n_baselines"],
            noise_std=xai_cfg["gradshap"]["noise_std"],
        )

    ids_to_process = image_ids[:smoke_n] if smoke_n else image_ids
    skipped = 0
    transform = get_val_transforms(cfg["data"]["image_size"])

    img_to_idx = {row["Image Index"]: i for i, row in dataset.labels_df.iterrows()}

    print(f"\n  [{combo_name}] Generating {len(ids_to_process)} heatmaps...")

    for image_id in tqdm(ids_to_process, desc=f"  {combo_name}", unit="img", ncols=90):
        out_path = out_dir / f"{Path(image_id).stem}.npy"

        # Skip already cached
        if out_path.exists():
            skipped += 1
            continue

        if image_id not in img_to_idx:
            continue

        img_tensor, target, *_ = dataset[img_to_idx[image_id]]

        try:
            if method == "gradcam":
                heatmap = explainer.explain(img_tensor)
            else:
                # Explain for the most confident positive class
                with torch.no_grad():
                    logits = model(img_tensor.unsqueeze(0).to(device))
                    probs  = torch.sigmoid(logits).squeeze()
                # Use argmax as explanation target
                class_idx = int(probs.argmax().item())
                heatmap = explainer.explain(img_tensor, class_idx=class_idx)

            np.save(str(out_path), heatmap)

        except Exception as e:
            print(f"\n  ⚠ Error on {image_id}: {e}")
            continue

    saved = len(list(out_dir.glob("*.npy")))
    print(f"  ✓ {combo_name}: {saved} heatmaps saved "
          f"({skipped} skipped — already existed)")


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device()

    splits_dir = Path(cfg["data"]["splits_dir"])
    images_dir = Path(cfg["data"]["images_dir"])

    print("=" * 60)
    print("  XAI Heatmap Generation")
    if args.smoke_test:
        print(f"  ⚡ SMOKE TEST: {cfg['smoke_test']['n_heatmap_images']} images")
    print("=" * 60)

    # ── Load test set (bbox images) ───────────────────────────────────
    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        print("❌ Run scripts/01_prepare_data.py first.")
        sys.exit(1)

    test_df  = pd.read_csv(test_csv)
    bbox_df  = pd.read_csv(splits_dir / "bbox_annotations.csv")
    test_ds  = ChestXrayDataset(
        images_dir, test_df, bbox_df=bbox_df,
        transform=get_val_transforms(cfg["data"]["image_size"]),
        return_bbox=True,
    )

    # Only process bbox-annotated images
    bbox_ids = list(bbox_df["Image Index"].unique())
    smoke_n  = cfg["smoke_test"]["n_heatmap_images"] if args.smoke_test else None

    print(f"\n  Test images with bounding boxes: {len(bbox_ids)}")

    # ── Determine which models and methods to run ─────────────────────
    model_names = ["resnet50", "efficientnet"] if args.model == "both" else [args.model]
    methods     = ["gradcam", "shap"]          if args.method == "both" else [args.method]

    for model_name in model_names:
        print(f"\n── Loading {model_name.upper()} ──")
        model = load_model(model_name, cfg, device)

        for method in methods:
            generate_for_combination(
                model=model,
                model_name=model_name,
                method=method,
                dataset=test_ds,
                image_ids=bbox_ids,
                cfg=cfg,
                device=device,
                smoke_n=smoke_n,
            )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Heatmap generation complete!")
    heatmaps_dir = Path(cfg["output"]["heatmaps_dir"])
    total = sum(len(list((heatmaps_dir / c).glob("*.npy")))
                for c in ["resnet50_gradcam", "resnet50_shap",
                           "efficientnet_gradcam", "efficientnet_shap"]
                if (heatmaps_dir / c).exists())
    print(f"  Total heatmaps cached: {total}")
    print(f"  Next: python scripts/05_compute_metrics.py")


if __name__ == "__main__":
    main()
