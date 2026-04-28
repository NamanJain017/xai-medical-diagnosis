import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

# Add current directory to path so src imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.utils.gpu_utils import get_device
from src.models.resnet50 import build_resnet50
from src.models.efficientnet import build_efficientnet
from src.data.transforms import get_val_transforms
from src.xai.gradcam import GradCAMExplainer
from src.xai.gradshap import GradientSHAPExplainer
from src.data.dataset import CLASSES
from src.evaluation.fidelity_metric import compute_fidelity_single
from src.evaluation.stability_metric import compute_stability_single

# --- CONFIGURATION ---
st.set_page_config(
    page_title="XAI Medical Diagnosis",
    page_icon="🩻",
    layout="wide"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 700;
    }
    h2 {
        color: #2563EB;
        font-weight: 600;
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Configuration & Device...")
def setup_env():
    cfg = load_config("config/config.yaml")
    device = get_device()
    return cfg, device

@st.cache_resource(show_spinner="Loading Models...")
def load_models(_cfg, _device):
    models = {}
    models_dir = Path(_cfg["output"]["models_dir"])
    num_classes = _cfg["num_classes"]
    
    # ResNet50
    resnet = build_resnet50(num_classes=num_classes, pretrained=False)
    resnet_ckpt = models_dir / "resnet50" / "best.pth"
    if resnet_ckpt.exists():
        ckpt = torch.load(resnet_ckpt, map_location=_device)
        resnet.load_state_dict(ckpt["model_state_dict"])
    resnet = resnet.to(_device).eval()
    models["ResNet50"] = resnet
    
    # EfficientNet
    effnet = build_efficientnet(num_classes=num_classes, pretrained=False)
    effnet_ckpt = models_dir / "efficientnet" / "best.pth"
    if effnet_ckpt.exists():
        ckpt = torch.load(effnet_ckpt, map_location=_device)
        effnet.load_state_dict(ckpt["model_state_dict"])
    effnet = effnet.to(_device).eval()
    models["EfficientNet"] = effnet
    
    return models

def overlay_heatmap(img_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    if heatmap.shape != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlaid = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid

# --- MAIN APP ---
def main():
    cfg, device = setup_env()
    models = load_models(cfg, device)
    transform = get_val_transforms(cfg["data"]["image_size"])
    
    st.title("🩻 XAI Medical Diagnosis")
    st.markdown("Upload a Chest X-ray image to get a diagnostic prediction and visualize the interpretability using Grad-CAM and SHAP.")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("Select Model", ["ResNet50", "EfficientNet", "Both Models"])
        xai_method = st.radio("XAI Method", ["Both", "Grad-CAM", "SHAP"])
        
        heatmap_alpha = st.slider("Heatmap Overlay Opacity", 0.1, 0.9, 0.5, 0.1)
        
        st.markdown("---")
        st.info("The models are trained to detect 14 different pathologies from chest X-rays. Grad-CAM and SHAP show which regions of the image drove the prediction.")
        st.caption("💡 **Tip:** EfficientNet generally performs slightly better overall (Avg AUC 0.838 vs ResNet50's 0.825). Comparing both models side-by-side provides a robust 'second opinion' for ambiguous cases.")

    # --- UPLOADER ---
    uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_tensor = transform(image)
        
        models_to_run = ["ResNet50", "EfficientNet"] if model_choice == "Both Models" else [model_choice]
        
        for m_name in models_to_run:
            if model_choice == "Both Models":
                st.markdown(f"<hr><h2 style='text-align: center;'>{m_name}</h2>", unsafe_allow_html=True)
                
            selected_model = models[m_name]
            
            # --- PREDICTION ---
            with st.spinner(f"Running inference with {m_name}..."):
                with torch.no_grad():
                    logits = selected_model(img_tensor.unsqueeze(0).to(device))
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Get top 3 predictions
            top_indices = probs.argsort()[-3:][::-1]
            top_class_idx = int(top_indices[0])
            
            st.subheader(f"Diagnostic Predictions" if model_choice != "Both Models" else f"Top Predictions ({m_name})")
            cols = st.columns(3)
            for i, idx in enumerate(top_indices):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="margin:0; color:#4B5563;">{CLASSES[idx]}</h4>
                        <h2 style="margin:0; color:#2563EB;">{probs[idx]:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # --- XAI EXPLANATIONS ---
            st.markdown("---")
            st.subheader(f"Interpretability for Top Prediction: **{CLASSES[top_class_idx]}**")
            
            # Generate Explanations
            gradcam_hm, shap_hm = None, None
            gcam_metrics, shap_metrics = {}, {}
            
            with st.spinner(f"Generating explanations and computing metrics for {m_name}... (This may take a moment)"):
                if xai_method in ["Grad-CAM", "Both"]:
                    target_layer = selected_model.get_gradcam_target_layer()
                    gcam_explainer = GradCAMExplainer(selected_model, target_layer, device)
                    gradcam_hm = gcam_explainer.explain(img_tensor)
                    
                    # Metrics
                    fid = compute_fidelity_single(selected_model, img_tensor, gradcam_hm, top_class_idx, [20.0], device)
                    gcam_metrics['fidelity'] = fid['fidelity_per_k'][20.0] * 100
                    stab = compute_stability_single(img_tensor, lambda x: gcam_explainer.explain(x), gradcam_hm, n_samples=3)
                    gcam_metrics['stability'] = stab['mean_cosine_sim']
                    
                if xai_method in ["SHAP", "Both"]:
                    shap_explainer = GradientSHAPExplainer(
                        selected_model, device,
                        n_baselines=cfg["xai"]["gradshap"]["n_baselines"],
                        noise_std=cfg["xai"]["gradshap"]["noise_std"]
                    )
                    shap_hm = shap_explainer.explain(img_tensor, class_idx=top_class_idx)
                    
                    # Metrics
                    fid = compute_fidelity_single(selected_model, img_tensor, shap_hm, top_class_idx, [20.0], device)
                    shap_metrics['fidelity'] = fid['fidelity_per_k'][20.0] * 100
                    stab = compute_stability_single(img_tensor, lambda x: shap_explainer.explain(x, class_idx=top_class_idx), shap_hm, n_samples=3)
                    shap_metrics['stability'] = stab['mean_cosine_sim']

            # Display
            display_cols = 1
            if xai_method == "Both":
                display_cols = 3
            elif xai_method in ["Grad-CAM", "SHAP"]:
                display_cols = 2
                
            columns = st.columns(display_cols)
            
            with columns[0]:
                st.image(image, caption="Original X-ray", width="stretch")
                
            col_idx = 1
            if gradcam_hm is not None:
                with columns[col_idx]:
                    overlaid_gcam = overlay_heatmap(img_np, gradcam_hm, alpha=heatmap_alpha)
                    st.image(overlaid_gcam, caption=f"Grad-CAM ({m_name})", width="stretch")
                    st.caption(f"📉 **Fidelity (20% mask):** -{gcam_metrics['fidelity']:.1f}% drop  \n🛡️ **Stability (Sim):** {gcam_metrics['stability']:.2f}")
                col_idx += 1
                
            if shap_hm is not None:
                with columns[col_idx]:
                    overlaid_shap = overlay_heatmap(img_np, shap_hm, alpha=heatmap_alpha)
                    st.image(overlaid_shap, caption=f"SHAP ({m_name})", width="stretch")
                    st.caption(f"📉 **Fidelity (20% mask):** -{shap_metrics['fidelity']:.1f}% drop  \n🛡️ **Stability (Sim):** {shap_metrics['stability']:.2f}")

if __name__ == "__main__":
    main()
