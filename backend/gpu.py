"""GPU detection via nvidia-smi."""
import subprocess
from typing import Optional


# VRAM estimates in GB for known Ollama model tags
MODEL_VRAM: dict[str, float] = {
    "qwen2.5:7b": 4.5,
    "qwen2.5:14b": 8.5,
    "qwen2.5:32b": 18.0,
    "qwen2.5-coder:7b": 4.5,
    "qwen2.5-coder:14b": 8.5,
    "dolphin-llama3:8b": 5.0,
    "dolphin-phi": 2.5,
    "dolphin-phi:latest": 2.5,
    "llama3.2:3b": 2.5,
    "llama3.2:1b": 1.5,
    "llama3:8b": 5.0,
    "mistral:7b": 4.5,
    "mistral:latest": 4.5,
    "phi3:mini": 2.5,
    "phi3:medium": 8.0,
    "gemma2:9b": 5.5,
    "gemma2:2b": 2.0,
}

# Fallback: estimate from parameter count in name
def _estimate_vram(model_name: str) -> float:
    name = model_name.lower()
    if "70b" in name: return 40.0
    if "34b" in name: return 20.0
    if "32b" in name: return 18.0
    if "14b" in name: return 8.5
    if "13b" in name: return 8.0
    if "8b" in name:  return 5.0
    if "7b" in name:  return 4.5
    if "3b" in name:  return 2.5
    if "2b" in name:  return 2.0
    if "1b" in name:  return 1.5
    return 4.5  # safe default


def vram_for_model(model_name: str) -> float:
    if model_name in MODEL_VRAM:
        return MODEL_VRAM[model_name]
    # strip tag and try base name
    base = model_name.split(":")[0] + ":latest"
    if base in MODEL_VRAM:
        return MODEL_VRAM[base]
    return _estimate_vram(model_name)


def detect_gpu() -> Optional[dict]:
    """Run nvidia-smi and return GPU info, or None if no NVIDIA GPU found."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None
        parts = [p.strip() for p in lines[0].split(",")]
        return {
            "name": parts[0],
            "vram_total_gb": round(int(parts[1]) / 1024, 1),
            "vram_used_gb": round(int(parts[2]) / 1024, 1),
            "vram_free_gb": round(int(parts[3]) / 1024, 1),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def check_model_fit(models: list[str], gpu: Optional[dict]) -> dict:
    """Given a list of model names and GPU info, return fit analysis."""
    total_vram = sum(vram_for_model(m) for m in models)
    gpu_vram = gpu["vram_total_gb"] if gpu else 0.0

    fits = gpu_vram > 0 and total_vram <= gpu_vram
    per_model = [
        {"model": m, "vram_gb": vram_for_model(m)} for m in models
    ]
    return {
        "total_vram_needed_gb": round(total_vram, 1),
        "gpu_vram_gb": gpu_vram,
        "fits_simultaneously": fits,
        "per_model": per_model,
        "warning": (
            None if fits or not models
            else f"Selected models need {total_vram:.1f} GB but GPU has {gpu_vram:.1f} GB. "
                 "Agents will be scheduled to run at different times."
        ),
    }
