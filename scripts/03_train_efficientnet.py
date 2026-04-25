"""
Script 03: Train EfficientNet-B0 on NIH Chest X-ray14.

Usage:
  python scripts/03_train_efficientnet.py
  python scripts/03_train_efficientnet.py --smoke-test
  python scripts/03_train_efficientnet.py --resume

After training:
  - Best checkpoint: outputs/models/efficientnet/best.pth
  - Training log:    outputs/logs/train_efficientnet.csv
  - TensorBoard:     outputs/logs/tensorboard/efficientnet/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.class_weights import load_class_weights
from src.models.efficientnet import build_efficientnet
from src.training.trainer import Trainer
from src.training.metrics import collect_predictions, compute_auc
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.gpu_utils import get_device, print_gpu_memory


def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 on NIH Chest X-ray14")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device()

    splits_dir = Path(cfg["data"]["splits_dir"])
    images_dir = Path(cfg["data"]["images_dir"])
    img_size   = cfg["data"]["image_size"]
    train_cfg  = cfg["training"]["efficientnet"]

    train_csv = splits_dir / "train.csv"
    val_csv   = splits_dir / "val.csv"
    if not train_csv.exists():
        print("❌ Split files not found. Run scripts/01_prepare_data.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    if args.smoke_test:
        n = cfg["smoke_test"]["n_images"]
        train_df = train_df.sample(min(n, len(train_df)), random_state=42)
        val_df   = val_df.sample(min(n // 5, len(val_df)), random_state=42)
        train_cfg = {**train_cfg, "epochs": cfg["smoke_test"]["epochs"]}
        cfg["training"]["efficientnet"] = train_cfg
        print(f"⚡ SMOKE TEST: {len(train_df)} train / {len(val_df)} val images, "
              f"{train_cfg['epochs']} epochs")

    train_ds = ChestXrayDataset(images_dir, train_df, transform=get_train_transforms(img_size))
    val_ds   = ChestXrayDataset(images_dir, val_df,   transform=get_val_transforms(img_size))

    n_workers = cfg["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],  # 32 — EfficientNet is lighter
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=ChestXrayDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=ChestXrayDataset.collate_fn,
    )

    print(f"\n  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print_gpu_memory()

    weights_path = splits_dir / "class_weights.npy"
    pos_weight = load_class_weights(str(weights_path)) if weights_path.exists() else None

    model = build_efficientnet(num_classes=cfg["num_classes"], pretrained=True)

    if args.resume:
        ckpt_path = Path(cfg["output"]["models_dir"]) / "efficientnet" / "latest.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  ↩ Resumed from {ckpt_path} (epoch {ckpt['epoch']})")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        model_name="efficientnet",
        device=device,
        pos_weight=pos_weight,
        output_dir=cfg["output"]["models_dir"].replace("/models", ""),
    )
    trainer.train()

    print("\n── Final Evaluation on Validation Set ──")
    best_ckpt = Path(cfg["output"]["models_dir"]) / "efficientnet" / "best.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    targets, probs = collect_predictions(model, val_loader, device)
    compute_auc(targets, probs, verbose=True)

    print(f"\n✅ EfficientNet-B0 training complete!")
    print(f"   Next: python scripts/04_generate_heatmaps.py")


if __name__ == "__main__":
    main()
