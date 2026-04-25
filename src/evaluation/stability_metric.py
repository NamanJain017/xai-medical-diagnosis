"""
Metric 3: Explanation Stability (Perturbation Consistency)

Research question: Does the explanation remain consistent for visually similar inputs?

Method:
  1. Generate explanation E₁ for original image I
  2. Add imperceptible Gaussian noise (σ=0.02) to get I' (perturbed)
  3. Generate explanation E₂ for I'
  4. Stability = Cosine Similarity(E₁.flatten(), E₂.flatten())
  5. Average over N=5 noise samples per image

  High similarity (→1.0) = stable, trustworthy explanation
  Low similarity (→0.0)  = explanation is unreliable under tiny input variation

  This addresses the concern from Adebayo et al. (2018):
  "Sanity Checks for Saliency Maps" — many saliency methods fail even basic
  randomization tests, meaning their maps change unpredictably.
"""

import numpy as np
import torch
from typing import Callable, List


def add_gaussian_noise(
    img_tensor: torch.Tensor,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """
    Add imperceptible Gaussian noise to a normalized image tensor.

    Args:
        img_tensor: (C, H, W) normalized tensor
        noise_std:  Standard deviation of noise (σ=0.02 is imperceptible to humans)

    Returns:
        noisy_tensor: (C, H, W) perturbed image
    """
    noise = torch.randn_like(img_tensor) * noise_std
    return img_tensor + noise


def cosine_similarity_maps(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Compute cosine similarity between two explanation maps.

    Args:
        map1, map2: (H, W) numpy arrays

    Returns:
        cosine similarity in [-1, 1], typically close to [0, 1] for heatmaps
    """
    v1 = map1.flatten().astype(np.float64)
    v2 = map2.flatten().astype(np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0  # zero map → undefined, treat as unstable

    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_stability_single(
    img_tensor: torch.Tensor,
    explain_fn: Callable[[torch.Tensor], np.ndarray],
    original_explanation: np.ndarray,
    n_samples: int = 5,
    noise_std: float = 0.02,
) -> dict:
    """
    Compute stability score for a single image.

    Args:
        img_tensor:            (C, H, W) original normalized image tensor
        explain_fn:            Function (img_tensor) → (H, W) heatmap numpy array
        original_explanation:  Pre-computed E₁ for the original image
        n_samples:             Number of noisy variants to generate
        noise_std:             Noise level σ

    Returns:
        dict with 'mean_cosine_sim', 'std_cosine_sim', 'scores' (list)
    """
    similarities = []

    for _ in range(n_samples):
        noisy_img = add_gaussian_noise(img_tensor, noise_std=noise_std)
        noisy_explanation = explain_fn(noisy_img)
        sim = cosine_similarity_maps(original_explanation, noisy_explanation)
        similarities.append(sim)

    return {
        "mean_cosine_sim": float(np.mean(similarities)),
        "std_cosine_sim":  float(np.std(similarities)),
        "scores":          similarities,
    }


def aggregate_stability_scores(all_scores: List[dict]) -> dict:
    """
    Aggregate stability scores across all images.

    Returns:
        dict with overall mean and std
    """
    means = [s["mean_cosine_sim"] for s in all_scores]
    return {
        "mean": float(np.mean(means)) if means else float("nan"),
        "std":  float(np.std(means))  if means else float("nan"),
        "n":    len(means),
    }
