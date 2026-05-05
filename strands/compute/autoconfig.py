"""Auto-config helpers for the neural Compute Module.

Picks a model + batch size that fits the available hardware:

  >=16 GB VRAM    "large"   d=384 h=8 L=6 ff=1536 seq=2048 batch=16  ~25M params
  8-16 GB VRAM    "medium"  d=256 h=8 L=6 ff=1024 seq=2048 batch=16  ~9M params
  4-8 GB VRAM     "small"   d=192 h=6 L=4 ff=768  seq=1536 batch=12  ~3M params
  CPU / <4 GB     "tiny"    d=128 h=4 L=4 ff=512  seq=1024 batch=8   ~1M params

The "tiny" config is what we trained successfully on CPU. The larger
ones target a GPU; ROCm and CUDA share the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from strands.compute.model import ModelConfig


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware + recommended configuration."""
    device: str          # 'cuda' or 'cpu'
    device_name: str     # human-readable
    vram_gb: float       # 0 for CPU
    backend: str         # 'CUDA', 'ROCm', or 'CPU'
    profile: str         # 'large' | 'medium' | 'small' | 'tiny'
    model_config: ModelConfig
    batch_size: int


def detect_hardware() -> HardwareProfile:
    """Detect device and pick a sensible model/batch size."""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        backend = (
            "ROCm" if getattr(torch.version, "hip", None) is not None
            else "CUDA"
        )
        if vram_gb >= 16:
            profile = "large"
            model = ModelConfig(
                d_model=384, n_heads=8, n_layers=6, d_ff=1536,
                max_seq_len=2048, dropout=0.05,
            )
            batch = 16
        elif vram_gb >= 8:
            profile = "medium"
            model = ModelConfig(
                d_model=256, n_heads=8, n_layers=6, d_ff=1024,
                max_seq_len=2048, dropout=0.05,
            )
            batch = 16
        elif vram_gb >= 4:
            profile = "small"
            model = ModelConfig(
                d_model=192, n_heads=6, n_layers=4, d_ff=768,
                max_seq_len=1536, dropout=0.05,
            )
            batch = 12
        else:
            # Tiny GPU; same config as CPU.
            profile = "tiny"
            model = ModelConfig(
                d_model=128, n_heads=4, n_layers=4, d_ff=512,
                max_seq_len=1024, dropout=0.05,
            )
            batch = 8
    else:
        device = "cpu"
        device_name = "CPU"
        vram_gb = 0.0
        backend = "CPU"
        profile = "tiny"
        model = ModelConfig(
            d_model=128, n_heads=4, n_layers=4, d_ff=512,
            max_seq_len=1024, dropout=0.0,
        )
        batch = 8

    return HardwareProfile(
        device=device,
        device_name=device_name,
        vram_gb=vram_gb,
        backend=backend,
        profile=profile,
        model_config=model,
        batch_size=batch,
    )


def steps_for_profile(profile: str) -> int:
    """Reasonable max-step counts per profile. The orchestrator
    combines this with early stopping so we don't burn cycles past
    convergence."""
    return {
        "large":  20000,
        "medium": 15000,
        "small":  10000,
        "tiny":   5000,
    }.get(profile, 5000)


def warmup_for_profile(profile: str) -> int:
    return {
        "large":  500,
        "medium": 300,
        "small":  200,
        "tiny":   100,
    }.get(profile, 100)
