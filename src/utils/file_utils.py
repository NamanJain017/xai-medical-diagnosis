"""
File I/O utilities.
"""

import json
import yaml
from pathlib import Path

def ensure_dir(path: str | Path):
    """Ensure that a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: dict, path: str | Path):
    """Save a dictionary to a JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str | Path) -> dict:
    """Load a dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
