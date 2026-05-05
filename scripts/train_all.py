"""End-to-end training pipeline for the neural Compute Module.

One command does everything:

  1. Detect device (CUDA / ROCm / CPU)
  2. Pick a model + batch size that fits the hardware
  3. Build the backbone if missing
  4. Generate / refresh the training corpus
  5. Train with live progress + early stopping
  6. Run inference comparison + memorisation eval at the end

ROCm note: AMD GPUs through ROCm-built PyTorch present as
``torch.cuda.*``. Install the ROCm wheel (e.g.
``pip install --index-url https://download.pytorch.org/whl/rocm6.0
torch``) and this script picks it up automatically — no flags needed.

Usage:
    python scripts/train_all.py                         # full pipeline
    python scripts/train_all.py --quick                 # tiny model, 200 steps
    python scripts/train_all.py --steps 10000           # override max_steps
    python scripts/train_all.py --skip-capture          # use existing JSONL
    python scripts/train_all.py --device cpu            # force CPU even on GPU
    python scripts/train_all.py --multiturn-sessions 60 # bigger corpus
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

# Imports rely on the package being importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strands.compute.autoconfig import (
    detect_hardware,
    steps_for_profile,
    warmup_for_profile,
)
from strands.compute.model import ModelConfig
from strands.compute.train import TrainingConfig, train


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKBONE_DIR = REPO_ROOT / "strands" / "data" / "backbone"
CORPUS_PATH = REPO_ROOT / "strands" / "data" / "compute" / "train.jsonl"
CHECKPOINT_DIR = REPO_ROOT / "strands" / "data" / "compute" / "checkpoints"


# --- Pretty banners -----------------------------------------------------


def _banner(text: str) -> None:
    bar = "─" * max(60, len(text) + 4)
    print()
    print(bar)
    print(f"  {text}")
    print(bar)


def _kv(key: str, value: object) -> None:
    print(f"  {key:<22} {value}")


# --- Step 1: device + config -------------------------------------------


def _detect(device_override: str | None, profile_override: str | None,
            ) -> tuple[object, ModelConfig, int]:
    hw = detect_hardware()
    if device_override is not None:
        # Force a device; rebuild profile to match
        if device_override == "cpu" and hw.device != "cpu":
            from strands.compute.autoconfig import HardwareProfile
            hw = HardwareProfile(
                device="cpu", device_name="CPU (forced)",
                vram_gb=0.0, backend="CPU", profile="tiny",
                model_config=ModelConfig(
                    d_model=128, n_heads=4, n_layers=4, d_ff=512,
                    max_seq_len=1024, dropout=0.0,
                ),
                batch_size=8,
            )

    _banner("1. Device & model")
    _kv("Device", hw.device_name)
    _kv("Backend", hw.backend)
    if hw.vram_gb > 0:
        _kv("VRAM", f"{hw.vram_gb:.1f} GB")
    _kv("Profile", hw.profile)
    _kv("Model d_model", hw.model_config.d_model)
    _kv("    n_layers", hw.model_config.n_layers)
    _kv("    n_heads", hw.model_config.n_heads)
    _kv("    seq_len", hw.model_config.max_seq_len)
    _kv("Batch size", hw.batch_size)

    profile = profile_override or hw.profile
    return hw, hw.model_config, hw.batch_size


# --- Step 2: backbone --------------------------------------------------


def _ensure_backbone() -> None:
    _banner("2. Backbone")
    if (BACKBONE_DIR / "backbone.manifest.json").exists():
        # Quick stat
        import json
        manifest = json.loads(
            (BACKBONE_DIR / "backbone.manifest.json").read_text(),
        )
        _kv("Status", "✓ present")
        _kv("Nodes", f"{manifest['node_count']:,}")
        _kv("Edges", f"{manifest['edge_count']:,}")
        return
    print("  Backbone missing — building (may take ~90s)…")
    rc = subprocess.call(
        [sys.executable, "scripts/build_backbone.py"],
        cwd=REPO_ROOT,
    )
    if rc != 0:
        sys.exit(f"backbone build failed (exit {rc})")
    _kv("Status", "✓ built")


# --- Step 3: corpus ----------------------------------------------------


def _ensure_corpus(skip_capture: bool, multiturn_sessions: int,
                   floor: float) -> None:
    _banner("3. Training corpus")
    if skip_capture and CORPUS_PATH.exists():
        size_kb = CORPUS_PATH.stat().st_size / 1024
        n_lines = sum(1 for _ in CORPUS_PATH.open())
        _kv("Status", "✓ existing (skip-capture)")
        _kv("Records", n_lines)
        _kv("Size", f"{size_kb:.1f} KB")
        return
    print(f"  Capturing training data (multiturn-sessions={multiturn_sessions})…")
    cmd = [
        sys.executable, "scripts/capture_training_data.py",
        "--multiturn-sessions", str(multiturn_sessions),
        "--confidence-floor", str(floor),
    ]
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc != 0:
        sys.exit(f"corpus capture failed (exit {rc})")
    elapsed = time.time() - t0
    n_lines = sum(1 for _ in CORPUS_PATH.open())
    size_kb = CORPUS_PATH.stat().st_size / 1024
    _kv("Status", "✓ captured")
    _kv("Records", n_lines)
    _kv("Size", f"{size_kb:.1f} KB")
    _kv("Elapsed", f"{elapsed:.1f}s")


# --- Step 4: train -----------------------------------------------------


def _train(
    model_cfg: ModelConfig, batch_size: int, *,
    max_steps: int, profile: str, device: str | None, seed: int,
    patience: int,
) -> dict:
    _banner("4. Training")
    cfg = TrainingConfig(
        data_path=CORPUS_PATH,
        out_dir=CHECKPOINT_DIR,
        model=model_cfg,
        batch_size=batch_size,
        max_steps=max_steps,
        eval_every=max(50, max_steps // 50),
        sample_every=max(200, max_steps // 10),
        log_every=max(10, max_steps // 100),
        warmup_steps=warmup_for_profile(profile),
        lr=3e-4,
        seed=seed,
        device=device or "auto",
        patience=patience,
    )
    return train(cfg)


# --- Step 5: eval ------------------------------------------------------


def _eval(checkpoint: Path) -> None:
    _banner("5. Memorization eval (training-set fit)")
    if not checkpoint.exists():
        print("  no checkpoint produced — skipping eval")
        return
    rc = subprocess.call(
        [sys.executable, "scripts/eval_neural_module.py",
         "--checkpoint", str(checkpoint),
         "--n", "30",
         "--temperature", "0.0",
         "--max-new-tokens", "60"],
        cwd=REPO_ROOT,
    )
    if rc != 0:
        print("  eval failed (non-fatal)")


def _comparison(checkpoint: Path) -> None:
    _banner("6. Side-by-side: deterministic vs neural")
    if not checkpoint.exists():
        return
    rc = subprocess.call(
        [sys.executable, "scripts/test_neural_module.py",
         "--checkpoint", str(checkpoint),
         "--temperature", "0.7", "--top-k", "8",
         "--max-new-tokens", "60"],
        cwd=REPO_ROOT,
    )
    if rc != 0:
        print("  comparison failed (non-fatal)")


# --- Top-level ---------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end training pipeline for the neural "
                    "Compute Module.",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"],
                   default="auto",
                   help="Force a device (auto = cuda if available).")
    p.add_argument("--quick", action="store_true",
                   help="Tiny config + 200 steps for a smoke run.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override max steps. Default depends on profile.")
    p.add_argument("--skip-capture", action="store_true",
                   help="Reuse existing corpus instead of regenerating.")
    p.add_argument("--multiturn-sessions", type=int, default=60,
                   help="Number of multi-turn sessions in the corpus.")
    p.add_argument("--confidence-floor", type=float, default=0.99,
                   help="Force every turn through the seam (0.99) or "
                        "only low-confidence (0.5).")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stop if val doesn't improve in N evals.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-eval", action="store_true",
                   help="Skip the post-train eval/comparison.")
    args = p.parse_args()

    t_start = time.time()

    print("┌─ neural Compute Module — full training pipeline ─┐")
    _kv("PyTorch", torch.__version__)
    _kv("Has HIP (ROCm)", bool(getattr(torch.version, "hip", None)))
    _kv("CUDA available", torch.cuda.is_available())

    device_arg = None if args.device == "auto" else args.device
    hw, model_cfg, batch_size = _detect(device_arg, None)
    profile = hw.profile

    if args.quick:
        # Override for a fast smoke run.
        model_cfg = ModelConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            max_seq_len=1024, dropout=0.0,
        )
        batch_size = 4
        profile = "tiny"
        max_steps = args.steps or 200
    else:
        max_steps = args.steps or steps_for_profile(profile)

    _ensure_backbone()
    _ensure_corpus(
        args.skip_capture, args.multiturn_sessions, args.confidence_floor,
    )

    result = _train(
        model_cfg, batch_size,
        max_steps=max_steps, profile=profile,
        device=device_arg, seed=args.seed, patience=args.patience,
    )

    if not args.no_eval:
        ckpt = Path(result["best_checkpoint"])
        _eval(ckpt)
        _comparison(ckpt)

    _banner("Summary")
    _kv("Best val_loss", f"{result['best_val_loss']:.3f}")
    _kv("Steps actually run", result["actual_steps"])
    _kv("Early stopped", result["early_stopped"])
    _kv("Train wallclock", f"{result['elapsed_s']:.1f}s")
    _kv("Total wallclock", f"{time.time() - t_start:.1f}s")
    _kv("Checkpoint", result["best_checkpoint"])
    print()
    print("Try the model interactively:")
    print("  python scripts/test_neural_module.py "
          "--temperature 0.7 --top-k 8")
    print()


if __name__ == "__main__":
    main()
