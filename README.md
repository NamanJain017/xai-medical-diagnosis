# XAI Medical Diagnosis: Evaluating Explanation Fidelity in Chest X-ray CNNs

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.2.2-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a research-grade pipeline for Quantifying the Reliability and Faithfulness of AI explanations in CNN-based chest X-ray diagnosis. We compare **ResNet50** and **EfficientNet-B0** models using **Grad-CAM** and **GradientSHAP** across the NIH Chest X-ray14 dataset.

## 🚀 Key Results

| Metric | Highest Score | Combination |
| :--- | :--- | :--- |
| **Diagnostic Accuracy (AUC)** | **0.8387** | EfficientNet-B0 |
| **Explanation Fidelity** | **0.6424** | EfficientNet + SHAP |
| **Explanation Stability** | **0.9582** | ResNet50 + Grad-CAM |

### 🏥 Diagnostic Performance
The models achieved high diagnostic accuracy across 14 diseases. For simpler interpretation, the **EfficientNet-B0 model achieved a Mean Diagnostic Performance of 83.8%** (represented as AUC 0.838), with peak performance on Hernia (96.3%) and Cardiomegaly (91.2%).

## 🛠️ Project Structure
- `src/`: Core implementation of models (ResNet, EfficientNet), XAI methods (Grad-CAM, SHAP), and Metrics.
- `scripts/`: End-to-end execution pipeline from data prep to final analysis.
- `config/`: Centralized YAML configuration for hyperparameters and paths.
- `outputs/figures/`: Presentation-ready charts and visual heatmap overlays.

## 📈 XAI Evaluation Metrics
We go beyond "visual looking" heatmaps by mathematically quantifying:
1.  **IoU (Localization):** Overlap between AI focus and radiologist bounding boxes.
2.  **Fidelity (Deletion Test):** Measuring the drop in AI confidence when top-ranked pixels are removed.
3.  **Stability (Robustness):** Measuring heatmap consistency under input noise.

## 🖼️ Visual Examples
(Example heatmap overlays can be found in `outputs/figures/`)
The project includes tools to generate 1x5 comparison grids showing the Original X-ray (with doctor's bounding box) compared against four different AI explanation strategies.

## ⚙️ Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/xai-medical-diagnosis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download NIH Dataset and place in `data/raw/images`.

## 📜 Acknowledgments
- NIH Chest X-ray14 Dataset
- Captum (for SHAP implementation)
- Grad-CAM (for visualization)
