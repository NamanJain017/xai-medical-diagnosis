"""GPU/CUDA utilities."""
import torch


def get_device() -> torch.device:
    """Return the best available device with detailed info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"[gpu] Using GPU: {props.name}")
        print(f"[gpu] VRAM: {vram_gb:.1f} GB | CUDA: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("[gpu] ⚠ CUDA not available — using CPU (training will be slow)")
    return device


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[gpu] Memory: {allocated:.0f}MB allocated | "
              f"{reserved:.0f}MB reserved | {total:.0f}MB total")
