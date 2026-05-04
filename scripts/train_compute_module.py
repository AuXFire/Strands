"""Train the neural Compute Module on captured deterministic data.

Usage:
    python scripts/train_compute_module.py [--max-steps 500]

Defaults to a CPU-friendly config (~1.6M params, batch_size=8) that
fits in 4 cores / 15 GB RAM and trains a full curve in ~10 minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from strands.compute.model import ModelConfig
from strands.compute.train import TrainingConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("strands/data/compute/train.jsonl"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("strands/data/compute/checkpoints"))
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )
    cfg = TrainingConfig(
        data_path=args.data,
        out_dir=args.out_dir,
        model=model_cfg,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        log_every=args.log_every,
        lr=args.lr,
        warmup_steps=args.warmup,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
