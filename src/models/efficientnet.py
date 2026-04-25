"""
EfficientNet-B0 fine-tuned for NIH Chest X-ray14 multi-label classification.

Changes from standard EfficientNet-B0:
  - Final classifier: Linear(1280 → 14)  [no Sigmoid — BCEWithLogitsLoss]
  - Backbone freezing for initial epochs (configurable)
  - Grad-CAM hook on last MBConv block: model.features[-1]

EfficientNet-B0 is ~5× lighter than ResNet50 (~5.3M vs ~25M params)
and allows larger batch sizes on the same GPU.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetB0XRay(nn.Module):
    """
    EfficientNet-B0 adapted for 14-class multi-label chest X-ray classification.

    Grad-CAM target layer: self.backbone.features[-1]
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace the classifier head
        # Original: Sequential(Dropout, Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[efficientnet] Backbone frozen — only classifier head training")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[efficientnet] Full backbone unfrozen")

    def get_gradcam_target_layer(self):
        """Return the target layer for Grad-CAM (last MBConv block)."""
        return self.backbone.features[-1]

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_efficientnet(num_classes: int = 14, pretrained: bool = True) -> EfficientNetB0XRay:
    model = EfficientNetB0XRay(num_classes=num_classes, pretrained=pretrained)
    params = model.count_parameters()
    print(f"[efficientnet] Built EfficientNet-B0 | "
          f"Total params: {params['total']:,} | "
          f"Trainable: {params['trainable']:,}")
    return model
