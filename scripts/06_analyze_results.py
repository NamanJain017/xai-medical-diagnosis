"""
Script 06: Analyze results and generate all publication-quality figures.

Reads outputs/metrics/summary_table.csv and creates:
  1. 4×3 comparison bar chart (main result figure)
  2. Fidelity curves (K% removed vs confidence drop)
  3. Per-combination radar chart
  4. Sample heatmap overlay figures (from test set)

Usage:
  python scripts/06_analyze_results.py
  python scripts/06_analyze_results.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Analyze results and generate figures")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


# ── Consistent color scheme ────────────────────────────────────────────
COLORS = {
    "resnet50_gradcam":    "#3B82F6",  # blue
    "resnet50_shap":       "#60A5FA",  # light blue
    "efficientnet_gradcam":"#10B981",  # green
    "efficientnet_shap":   "#34D399",  # light green
}


def plot_comparison_bar(df: pd.DataFrame, out_dir: Path):
    """
    Main result figure: grouped bar chart for IoU, Fidelity@20%, Stability.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("#0F172A")

    metrics = [
        ("iou_mean",           "IoU Score\n(Localization Accuracy ↑)"),
        ("fidelity_20_mean",   "Fidelity Score @ 20%\n(Confidence Drop ↑)"),
        ("stability_mean",     "Stability Score\n(Cosine Similarity ↑)"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        combos = df["combination"].tolist()
        values = df[col].tolist()
        colors = [COLORS.get(c, "#94A3B8") for c in combos]

        bars = ax.bar(range(len(combos)), values, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.6)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, color="#E2E8F0", fontweight="bold")

        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(
            [c.replace("_", "\n") for c in combos],
            fontsize=7.5, color="#94A3B8"
        )
        ax.set_title(title, fontsize=10, color="#E2E8F0", pad=12, fontweight="bold")
        ax.set_ylim(0, max(values + [0.01]) * 1.2)
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#64748B")
        ax.spines[:].set_color("#334155")
        ax.yaxis.label.set_color("#94A3B8")

    # Legend
    patches = [mpatches.Patch(color=COLORS[c], label=c.replace("_", " + "))
               for c in COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("XAI Method Comparison — All 4 Combinations × 3 Metrics",
                 fontsize=13, color="#E2E8F0", fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = out_dir / "metric_comparison_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_fidelity_curves(df: pd.DataFrame, out_dir: Path):
    """
    Fidelity curves: K% pixels removed vs confidence drop.
    """
    k_values = [10, 20, 30, 50]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    for _, row in df.iterrows():
        combo  = row["combination"]
        color  = COLORS.get(combo, "#94A3B8")
        values = [row.get(f"fidelity_{k}_mean", np.nan) for k in k_values]

        ax.plot(k_values, values, marker="o", color=color, linewidth=2.5,
                markersize=7, label=combo.replace("_", " + "))
        ax.fill_between(k_values, values, alpha=0.08, color=color)

    ax.set_xlabel("K% Pixels Removed", fontsize=11, color="#94A3B8")
    ax.set_ylabel("Mean Confidence Drop (S₀ - S₁)", fontsize=11, color="#94A3B8")
    ax.set_title("Fidelity Curves — Confidence Drop vs Pixels Removed",
                 fontsize=12, color="#E2E8F0", fontweight="bold", pad=14)
    ax.set_xticks(k_values)
    ax.tick_params(colors="#64748B")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#475569")

    out_path = out_dir / "fidelity_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def print_latex_table(df: pd.DataFrame):
    """Print the 4×3 result table in LaTeX format for the paper."""
    print("\n── LaTeX Table (copy into your paper) ──────────────────────")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Combination & IoU ↑ & Fidelity@20\\% ↑ & Stability ↑ \\\\")
    print("\\hline")
    for _, row in df.iterrows():
        combo = row["combination"].replace("_", " + ")
        print(f"{combo} & "
              f"{row['iou_mean']:.3f}$\\pm${row['iou_std']:.3f} & "
              f"{row['fidelity_20_mean']:.3f} & "
              f"{row['stability_mean']:.3f}$\\pm${row['stability_std']:.3f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Quantitative comparison of XAI methods across three evaluation metrics.}")
    print("\\end{table}")


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / "summary_table.csv"
    if not summary_path.exists():
        print(f"❌ Summary table not found: {summary_path}")
        print("   Run: python scripts/05_compute_metrics.py first")
        sys.exit(1)

    df = pd.read_csv(summary_path)
    print(f"\n  Loaded results for {len(df)} combinations")
    print(df.to_string(index=False))

    print(f"\n── Generating Figures → {figures_dir} ──")
    plot_comparison_bar(df, figures_dir)
    plot_fidelity_curves(df, figures_dir)
    print_latex_table(df)

    print(f"\n✅ Analysis complete! All figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
