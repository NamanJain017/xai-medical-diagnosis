"""
Grad-CAM implementation using pytorch-grad-cam library.

Grad-CAM (Gradient-weighted Class Activation Mapping):
  1. Forward pass → get predictions
  2. Backprop gradient of predicted class score w.r.t. final conv layer
  3. Global average pool the gradients → channel weights
  4. Weighted sum of feature maps → coarse 7×7 heatmap
  5. ReLU + upsample to 224×224

Fast: ~milliseconds per image. No model modification needed.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExplainer:
    """
    Wraps pytorch-grad-cam's GradCAM for our multi-label classification setup.

    Args:
        model:        Trained PyTorch model (ResNet50XRay or EfficientNetB0XRay)
        target_layer: The convolutional layer to hook into
        device:       torch.device
    """

    def __init__(self, model, target_layer, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()

        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer],
        )

    @torch.no_grad()
    def get_predicted_class(self, img_tensor: torch.Tensor) -> int:
        """Return the class index with highest logit (for single image)."""
        logits = self.model(img_tensor.unsqueeze(0).to(self.device))
        return int(torch.argmax(logits.squeeze()).item())

    def explain(
        self,
        img_tensor: torch.Tensor,
        class_idx: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            img_tensor: (C, H, W) normalized tensor
            class_idx:  Disease class index to explain. If None, uses predicted class.

        Returns:
            heatmap: (H, W) numpy array normalized to [0, 1]
        """
        input_tensor = img_tensor.unsqueeze(0).to(self.device)

        if class_idx is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
            class_idx = int(torch.argmax(logits.squeeze()).item())

        targets = [ClassifierOutputTarget(class_idx)]

        # pytorch-grad-cam returns (1, H, W) normalized to [0,1] already
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
        )
        heatmap = grayscale_cam[0]  # (H, W)
        return heatmap.astype(np.float32)

    def explain_batch(
        self,
        img_tensors: torch.Tensor,
        class_indices: list = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Args:
            img_tensors:  (B, C, H, W) batch tensor
            class_indices: List of class indices (one per image). Uses argmax if None.

        Returns:
            heatmaps: (B, H, W) numpy array normalized to [0, 1]
        """
        B = img_tensors.shape[0]

        if class_indices is None:
            with torch.no_grad():
                logits = self.model(img_tensors.to(self.device))
            class_indices = torch.argmax(logits, dim=1).cpu().tolist()

        targets = [ClassifierOutputTarget(idx) for idx in class_indices]
        grayscale_cams = self.cam(
            input_tensor=img_tensors.to(self.device),
            targets=targets,
        )
        return grayscale_cams.astype(np.float32)  # (B, H, W)

    def __del__(self):
        # Clean up hooks
        if hasattr(self, "cam"):
            self.cam.__del__()
