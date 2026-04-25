"""
Heatmap visualization utilities.

Overlays normalized heatmaps on top of original X-ray images using colormaps.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        img_np: (H, W, 3) original image [0, 255] or [0, 1]
        heatmap: (H, W) normalized heatmap [0, 1]
        alpha: Transparency of the heatmap
        colormap: OpenCV colormap constant
        
    Returns:
        overlaid_img: (H, W, 3) image with heatmap overlay
    """
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid

def plot_side_by_side(img_np, heatmap, title="Explanation", save_path=None):
    """Plot original image and heatmap overlay side-by-side."""
    overlaid = overlay_heatmap(img_np, heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")
    
    axes[1].imshow(overlaid)
    axes[1].set_title(title)
    axes[1].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
