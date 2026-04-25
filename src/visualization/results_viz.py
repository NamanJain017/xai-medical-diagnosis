"""
Result visualization utilities.

Functions for creating bar charts, fidelity curves, and radar charts.
These are primarily used by the 06_analyze_results.py script.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# The actual plotting logic is mostly contained in scripts/06_analyze_results.py
# This file is reserved for any additional custom visualization utilities you may need.

def plot_per_disease_metrics(df: pd.DataFrame, metric: str, out_path: Path):
    """
    Optional utility to plot metrics broken down by disease class.
    Requires per-class metric data.
    """
    pass
