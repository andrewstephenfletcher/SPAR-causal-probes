"""Shared helpers: device selection, environment logging, cache clearing."""

import gc
import json
import platform

import torch
import transformers

from .config import Config


# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS > CPU
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_map():
    """
    Return device_map argument for AutoModelForCausalLM.from_pretrained.

    - CUDA: return "auto" (handles single-GPU and multi-GPU automatically).
    - MPS / CPU: return None — the caller is responsible for .to(device).
      (accelerate's device_map="auto" does not support MPS reliably.)
    """
    if torch.cuda.is_available():
        return "auto"
    return None


def clear_device_cache() -> None:
    """Free CUDA or MPS memory and run Python GC."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def gpu_memory_gb() -> float:
    """
    Return free accelerator memory in GB.

    - CUDA: uses torch.cuda.mem_get_info.
    - MPS (Apple Silicon unified memory): returns available system memory
      via psutil (the GPU and CPU share the same pool).
    - CPU-only: returns 0.
    """
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            # psutil not installed; assume plenty of RAM on Apple Silicon
            return 48.0
    return 0.0


# ---------------------------------------------------------------------------
# Environment logging
# ---------------------------------------------------------------------------

def log_environment(config: Config) -> dict:
    """Log environment details for reproducibility. Saves to output_dir/environment.json."""
    device = get_device()
    env_info: dict = {
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["gpu_count"] = torch.cuda.device_count()

    print("\nEnvironment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    env_path = config.output_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)

    return env_info
