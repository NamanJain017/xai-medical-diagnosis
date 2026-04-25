"""
Config loader utility.
"""
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
