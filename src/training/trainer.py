"""
Training loop for XAI Medical Diagnosis project.

Features:
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - Mixed precision training (torch.cuda.amp) — halves VRAM on RTX 4050
  - ReduceLROnPlateau scheduler
  - Early stopping on validation AUC
  - Per-epoch progress bars (tqdm)
  - TensorBoard logging
  - Checkpoint saving (best + latest)
  - Backbone freeze/unfreeze schedule
"""

import os
import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import collect_predictions, compute_auc


class Trainer:
    """
    Unified trainer for ResNet50 and EfficientNet-B0.

    Args:
        model:           PyTorch model (ResNet50XRay or EfficientNetB0XRay)
        train_loader:    Training DataLoader
        val_loader:      Validation DataLoader
        config:          Full config dict (from config.yaml)
        model_name:      'resnet50' or 'efficientnet'
        device:          torch.device
        output_dir:      Where to save checkpoints and logs
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        model_name: str,
        device: torch.device,
        pos_weight: Optional[torch.Tensor] = None,
        output_dir: str = "./outputs",
    ):
        self.model      = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config     = config
        self.model_name = model_name
        self.device     = device

        # ── Training config for this model ─────────────────────────────
        self.train_cfg = config["training"][model_name]
        self.epochs    = self.train_cfg["epochs"]
        self.freeze_epochs = self.train_cfg.get("freeze_epochs", 5)

        # ── Loss ──────────────────────────────────────────────────────
        pw = pos_weight.to(device) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # ── Optimizer ─────────────────────────────────────────────────
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # ── Scheduler ─────────────────────────────────────────────────
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config["training"]["scheduler_factor"],
            patience=config["training"]["scheduler_patience"],
            min_lr=config["training"]["min_lr"],
            verbose=True,
        )

        # ── AMP scaler ────────────────────────────────────────────────
        self.use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── Paths ─────────────────────────────────────────────────────
        self.model_dir = Path(output_dir) / "models" / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_path  = Path(output_dir) / "logs" / f"train_{model_name}.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        tb_dir = Path(output_dir) / "logs" / "tensorboard" / model_name
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # ── State ─────────────────────────────────────────────────────
        self.best_val_auc  = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.train_cfg["early_stopping_patience"]

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"  Train Epoch {epoch}", leave=False,
                    unit="batch", ncols=90)

        for imgs, targets, *_ in pbar:
            imgs    = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(imgs)
                loss   = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / n_batches

    def _val_epoch(self, epoch: int) -> tuple:
        targets_arr, probs_arr = collect_predictions(self.model, self.val_loader, self.device)
        result = compute_auc(targets_arr, probs_arr, verbose=False)
        return result["mean"], result["per_class"]

    def train(self) -> dict:
        """
        Run full training.
        Returns training history dict.
        """
        history = []

        # Write CSV header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_auc", "lr", "elapsed_sec"])

        print(f"\n{'═'*60}")
        print(f"  Training {self.model_name.upper()}")
        print(f"  Epochs: {self.epochs} | AMP: {self.use_amp} | "
              f"Freeze epochs: {self.freeze_epochs}")
        print(f"  Device: {self.device}")
        print(f"{'═'*60}\n")

        # Initial freeze
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # ── Unfreeze backbone after freeze_epochs ─────────────────
            if epoch == self.freeze_epochs + 1:
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                # Re-create optimizer with all params
                self.optimizer = Adam(
                    self.model.parameters(),
                    lr=self.train_cfg["learning_rate"],
                    weight_decay=self.train_cfg["weight_decay"],
                )

            train_loss = self._train_epoch(epoch)
            val_auc, per_class = self._val_epoch(epoch)
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_auc)

            # ── Logging ───────────────────────────────────────────────
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("AUC/val_mean", val_auc, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            row = {
                "epoch": epoch, "train_loss": train_loss,
                "val_auc": val_auc, "lr": lr, "elapsed_sec": round(elapsed, 1)
            }
            history.append(row)

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.4f}", f"{val_auc:.4f}",
                                  f"{lr:.2e}", f"{elapsed:.1f}"])

            auc_symbol = "✓" if val_auc >= 0.80 else "·"
            print(f"  Epoch {epoch:02d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} {auc_symbol} | "
                  f"LR: {lr:.1e} | "
                  f"Time: {elapsed:.1f}s")

            # ── Checkpoint ────────────────────────────────────────────
            self._save_checkpoint(epoch, val_auc, is_best=(val_auc > self.best_val_auc))

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # ── Early stopping ────────────────────────────────────────
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n  ⚠ Early stopping triggered at epoch {epoch} "
                      f"(no improvement for {self.early_stopping_patience} epochs)")
                break

        self.writer.close()
        print(f"\n  ✅ Training complete. Best Val AUC: {self.best_val_auc:.4f}")
        print(f"  Checkpoints saved to: {self.model_dir}\n")
        return history

    def _save_checkpoint(self, epoch: int, val_auc: float, is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auc": val_auc,
        }
        torch.save(ckpt, self.model_dir / "latest.pth")
        if is_best:
            torch.save(ckpt, self.model_dir / "best.pth")
            print(f"    💾 New best saved (AUC={val_auc:.4f})")
