"""
ResNet50 fine-tuned for NIH Chest X-ray14 multi-label classification.

Changes from standard ResNet50:
  - Final FC: Linear(2048 → 14)  [no Sigmoid — BCEWithLogitsLoss handles it]
  - Backbone freezing for initial epochs (configurable)
  - Grad-CAM hook registered on layer4[-1]
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50XRay(nn.Module):
    """
    ResNet50 adapted for 14-class multi-label chest X-ray classification.

    Grad-CAM target layer: self.backbone.layer4[-1]
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original FC head — keep everything up to avgpool
        self.backbone = backbone
        in_features = backbone.fc.in_features  # 2048

        # Replace with our 14-class head (no activation — BCEWithLogitsLoss)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the final FC head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[resnet50] Backbone frozen — only FC head training")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[resnet50] Full backbone unfrozen")

    def get_gradcam_target_layer(self):
        """Return the target layer for Grad-CAM (layer4's last block)."""
        return self.backbone.layer4[-1]

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_resnet50(num_classes: int = 14, pretrained: bool = True) -> ResNet50XRay:
    model = ResNet50XRay(num_classes=num_classes, pretrained=pretrained)
    params = model.count_parameters()
    print(f"[resnet50] Built ResNet50 | "
          f"Total params: {params['total']:,} | "
          f"Trainable: {params['trainable']:,}")
    return model
