"""
GradientSHAP implementation using Captum (Meta's XAI library).

GradientSHAP:
  1. Define a baseline distribution (random Gaussian noise images)
  2. For each image, randomly sample a baseline and a noise level α ∈ [0,1]
  3. Compute expected gradient of output w.r.t. input along the path baseline → input
  4. Multiply by (input - baseline) → pixel-level SHAP attribution
  5. Take absolute values and normalize to [0, 1] for comparison with Grad-CAM

Slower than Grad-CAM (10-20×) but more theoretically grounded (satisfies Shapley axioms).
MUST precompute and cache results — never recompute on the fly.
"""

import numpy as np
import torch
from captum.attr import GradientShap


class GradientSHAPExplainer:
    """
    GradientSHAP attribution using Captum.

    Args:
        model:       Trained PyTorch model
        device:      torch.device
        n_baselines: Number of random noise baseline images to use
        noise_std:   Standard deviation for baseline noise (σ=0.02 of image range)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_baselines: int = 20,
        noise_std: float = 0.02,
    ):
        self.model       = model
        self.device      = device
        self.n_baselines = n_baselines
        self.noise_std   = noise_std
        self.model.eval()

        self.explainer = GradientShap(model)

    def _make_baselines(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Create n_baselines random Gaussian noise images.
        Shape: (n_baselines, C, H, W)
        """
        baselines = torch.randn(
            self.n_baselines, *img_tensor.shape,
            device=self.device
        ) * self.noise_std
        return baselines

    def explain(
        self,
        img_tensor: torch.Tensor,
        class_idx: int = None,
    ) -> np.ndarray:
        """
        Generate GradientSHAP attribution map for a single image.

        Args:
            img_tensor: (C, H, W) normalized tensor
            class_idx:  Disease class index to explain. If None, uses argmax.

        Returns:
            attribution: (H, W) numpy array, absolute SHAP values normalized to [0, 1]
        """
        img_tensor = img_tensor.to(self.device)
        input_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)

        if class_idx is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
            class_idx = int(torch.argmax(logits.squeeze()).item())

        baselines = self._make_baselines(img_tensor)  # (n_baselines, C, H, W)

        # Captum GradientShap requires stochastic_sources for multiple baselines
        attr = self.explainer.attribute(
            inputs=input_tensor,
            baselines=baselines,
            target=class_idx,
            n_samples=self.n_baselines,
            stdevs=self.noise_std,
        )  # shape: (1, C, H, W)

        # Aggregate over channels, take absolute value
        attr = attr.squeeze(0)  # (C, H, W)
        attr = attr.abs().mean(dim=0)  # (H, W) — mean across channels

        attr_np = attr.detach().cpu().numpy()

        # Normalize to [0, 1]
        a_min, a_max = attr_np.min(), attr_np.max()
        if a_max > a_min:
            attr_np = (attr_np - a_min) / (a_max - a_min)
        else:
            attr_np = np.zeros_like(attr_np)

        return attr_np.astype(np.float32)

    def explain_with_multiple_targets(
        self,
        img_tensor: torch.Tensor,
        class_indices: list,
    ) -> dict:
        """
        Generate GradientSHAP maps for multiple disease classes on one image.
        Returns dict: {class_idx: attribution_map}
        """
        results = {}
        for cls_idx in class_indices:
            results[cls_idx] = self.explain(img_tensor, class_idx=cls_idx)
        return results
