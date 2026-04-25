import os
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration
RAW_IMAGES_DIR = Path("data/raw/images")
BBOX_CSV = Path("data/splits/bbox_annotations.csv")
HEATMAPS_DIR = Path("outputs/heatmaps")
FIGURES_DIR = Path("outputs/figures")

COMBINATIONS = [
    ("resnet50_gradcam", "ResNet50\nGrad-CAM"),
    ("resnet50_shap", "ResNet50\nSHAP"),
    ("efficientnet_gradcam", "EfficientNet\nGrad-CAM"),
    ("efficientnet_shap", "EfficientNet\nSHAP")
]

# Hardcoded data for standalone charting without needing to reload the entire dataset
CLASS_COUNTS = {
    "Atelectasis": 11559, "Cardiomegaly": 2776, "Effusion": 13317, "Infiltration": 19894,
    "Mass": 5782, "Nodule": 6331, "Pneumonia": 1431, "Pneumothorax": 5302,
    "Consolidation": 4667, "Edema": 2303, "Emphysema": 2516, "Fibrosis": 1686,
    "Pleural_Thickening": 3385, "Hernia": 227
}

RESNET_AUC = {
    "Atelectasis": 0.7893, "Cardiomegaly": 0.9060, "Effusion": 0.8852, "Infiltration": 0.7027,
    "Mass": 0.8444, "Nodule": 0.7748, "Pneumonia": 0.7210, "Pneumothorax": 0.8716,
    "Consolidation": 0.8091, "Edema": 0.8785, "Emphysema": 0.9260, "Fibrosis": 0.7726,
    "Pleural_Thickening": 0.7902, "Hernia": 0.8859
}

EFFICIENTNET_AUC = {
    "Atelectasis": 0.8031, "Cardiomegaly": 0.9121, "Effusion": 0.8814, "Infiltration": 0.7163,
    "Mass": 0.8430, "Nodule": 0.7746, "Pneumonia": 0.7174, "Pneumothorax": 0.8918,
    "Consolidation": 0.8171, "Edema": 0.8862, "Emphysema": 0.9280, "Fibrosis": 0.8147,
    "Pleural_Thickening": 0.7929, "Hernia": 0.9637
}

def overlay_heatmap(img_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays a 0-1 normalized heatmap onto a 0-255 RGB image."""
    # Ensure image is uint8
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
    # Resize heatmap to match image if necessary
    if heatmap.shape != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
    # Normalize heatmap just in case
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlaid = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def generate_heatmap_grids(n_samples=5, target_diseases=None, prefix="heatmap_example"):
    """Generates a 1x5 grid comparing original bbox and all 4 heatmaps for random images."""
    print(f"\n── Generating {n_samples} Heatmap Comparison Grids ──")
    
    bbox_df = pd.read_csv(BBOX_CSV)
    if target_diseases:
        bbox_df = bbox_df[bbox_df["Finding Label"].isin(target_diseases)]
        
    # Get unique image indices that have valid heatmaps in all 4 folders
    image_ids = bbox_df["Image Index"].unique()
    
    valid_ids = []
    for img_id in image_ids:
        # Check if heatmap exists in all 4 combos
        exists = all((HEATMAPS_DIR / combo / f"{Path(img_id).stem}.npy").exists() for combo, _ in COMBINATIONS)
        if exists:
            valid_ids.append(img_id)
            
    np.random.seed(42)  # For reproducible examples
    sampled_ids = np.random.choice(valid_ids, size=min(n_samples, len(valid_ids)), replace=False)
    
    for i, img_id in enumerate(sampled_ids):
        # 1. Load Original Image
        img_path = RAW_IMAGES_DIR / img_id
        if not img_path.exists():
            continue
            
        # Original image in original resolution
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = orig_img.shape[:2]
        
        # 2. Draw Bounding Boxes
        # An image might have multiple bounding boxes
        boxes = bbox_df[bbox_df["Image Index"] == img_id]
        img_with_bbox = orig_img.copy()
        
        disease_names = []
        for _, row in boxes.iterrows():
            disease_names.append(row["Finding Label"])
            x = float(row["Bbox [x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h]"])
            
            # Draw green rectangle (thickness=5)
            cv2.rectangle(img_with_bbox, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 8)
            
        disease_title = " | ".join(set(disease_names))
        
        # 3. Create the 1x5 Plot
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        fig.suptitle(f"Image: {img_id}   |   Ground Truth: {disease_title}", fontsize=16, y=1.05)
        
        # Plot A: Original + Bbox
        axes[0].imshow(img_with_bbox)
        axes[0].set_title("Original X-ray\n(Ground Truth Bbox)", fontsize=14)
        axes[0].axis("off")
        
        # Plot B-E: Overlays
        for idx, (combo_folder, combo_title) in enumerate(COMBINATIONS):
            hm_path = HEATMAPS_DIR / combo_folder / f"{Path(img_id).stem}.npy"
            heatmap_224 = np.load(str(hm_path))
            
            # Overlay directly onto the original high-res image (it resizes the heatmap automatically)
            overlaid = overlay_heatmap(orig_img, heatmap_224, alpha=0.5)
            
            axes[idx+1].imshow(overlaid)
            axes[idx+1].set_title(combo_title, fontsize=14)
            axes[idx+1].axis("off")
            
        plt.tight_layout()
        save_path = FIGURES_DIR / f"{prefix}_{i+1}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ Saved: {save_path}")


def generate_dataset_chart():
    """Generates a bar chart showing the class distribution."""
    print("\n── Generating Dataset Distribution Chart ──")
    
    # Sort by count descending
    sorted_classes = sorted(CLASS_COUNTS.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0].replace("_", " ") for x in sorted_classes]
    values = [x[1] for x in sorted_classes]
    
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x=labels, y=values, palette="viridis")
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("NIH Chest X-ray14: Disease Class Distribution", fontsize=18, pad=20)
    plt.ylabel("Number of Positive Images", fontsize=14)
    plt.xlabel("Disease Class", fontsize=14)
    
    # Add exact numbers on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.tight_layout()
    save_path = FIGURES_DIR / "dataset_distribution.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def generate_auc_comparison():
    """Generates a grouped bar chart comparing AUC scores."""
    print("\n── Generating Model AUC Comparison Chart ──")
    
    # Prepare DataFrame
    data = []
    for cls in RESNET_AUC.keys():
        clean_cls = cls.replace("_", " ")
        data.append({"Class": clean_cls, "AUC": RESNET_AUC[cls], "Model": "ResNet50 (Avg: 0.825)"})
        data.append({"Class": clean_cls, "AUC": EFFICIENTNET_AUC[cls], "Model": "EfficientNet-B0 (Avg: 0.838)"})
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=df, x="Class", y="AUC", hue="Model", palette=["#3498db", "#e74c3c"])
    
    # Add a dashed line for random guessing
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Random Guessing (0.50)')
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.ylim(0.4, 1.05) # Zoom in on the relevant range
    
    plt.title("Validation AUC by Disease Class", fontsize=18, pad=20)
    plt.ylabel("Area Under ROC Curve (AUC)", fontsize=14)
    plt.xlabel("")
    
    plt.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "model_auc_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Generate 5 examples specifically for the highest-performing models
    reliable_diseases = ["Hernia", "Emphysema", "Cardiomegaly", "Pneumothorax", "Edema"]
    generate_heatmap_grids(n_samples=5, target_diseases=reliable_diseases, prefix="reliable_heatmap")
    
    generate_dataset_chart()
    generate_auc_comparison()
    print("\n✅ All presentation visuals generated successfully!")
