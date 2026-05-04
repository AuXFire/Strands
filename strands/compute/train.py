"""Training loop for the neural Compute Module.

Designed for CPU-friendly test runs:
  - Tiny model (~1.6M params)
  - AdamW with linear warmup + cosine decay
  - Eval every N steps with masked-CE loss on a held-out split
  - Checkpoint best val loss
  - Sample generations periodically so you can watch behavior emerge

This is the "test training loop" — small dataset, small model, prove
the pipeline learns the deterministic mapping. Once it works,
upscale model + data and the same loop runs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from strands.compute.dataset import (
    ConditioningDataset,
    TrainingRecord,
    iterate_batches,
    split,
)
from strands.compute.model import ModelConfig, TinyTransformer
from strands.compute.tokenizer import EOS_ID, decode, encode_context


@dataclass
class TrainingConfig:
    data_path: Path = Path("strands/data/compute/train.jsonl")
    out_dir: Path = Path("strands/data/compute/checkpoints")
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 8
    max_steps: int = 500
    eval_every: int = 50
    sample_every: int = 100
    lr: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    val_fraction: float = 0.1
    seed: int = 0
    log_every: int = 10
    grad_clip: float = 1.0


def lr_at_step(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup → cosine decay to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def compute_loss(
    model: TinyTransformer, batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Standard causal-LM loss masked to target tokens only."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    logits = model(input_ids)  # (B, T, V)
    # Shift: predict labels[i+1] from input[i].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(
    model: TinyTransformer, val_records: list[TrainingRecord], *,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in iterate_batches(val_records, batch_size=batch_size, shuffle=False):
        loss = compute_loss(model, batch)
        total_loss += float(loss)
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


@torch.no_grad()
def sample_generation(
    model: TinyTransformer, context: str, *,
    max_new_tokens: int = 96, temperature: float = 0.0,
) -> str:
    """Greedy decode from a context string. Returns the generated
    target portion (everything after [SEP])."""
    model.eval()
    ids = torch.tensor([encode_context(context)], dtype=torch.long)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, eos_id=EOS_ID,
        temperature=temperature,
    )
    # Strip the context portion.
    generated_ids = out[0, ids.shape[1]:].tolist()
    return decode(generated_ids)


def train(cfg: TrainingConfig) -> dict:
    """Run training loop. Returns a dict with final metrics + best
    checkpoint path."""
    print(f"Loading dataset from {cfg.data_path} …")
    ds = ConditioningDataset(cfg.data_path, max_seq_len=cfg.model.max_seq_len)
    train_records, val_records = split(
        ds, val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    print(
        f"  {len(train_records)} train / {len(val_records)} val records"
    )

    print(f"Building model: {cfg.model}")
    torch.manual_seed(cfg.seed)
    model = TinyTransformer(cfg.model)
    print(f"  {model.num_parameters():,} parameters")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = cfg.out_dir / "best.pt"

    sample_contexts = [
        "[PROMPT] What is a cat? [INTENT] question_answering [QTYPE] definition [SEP]",
        "[PROMPT] Hello there [INTENT] social [QTYPE] greeting [SEP]",
    ]

    train_iter = _infinite_batches(
        train_records, batch_size=cfg.batch_size, seed=cfg.seed,
    )
    history: list[tuple[int, float, float | None]] = []

    t0 = time.time()
    model.train()
    for step in range(cfg.max_steps):
        batch = next(train_iter)
        for g in optim.param_groups:
            g["lr"] = lr_at_step(
                step, cfg.lr, cfg.warmup_steps, cfg.max_steps,
            )
        optim.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        loss_val = loss.detach().item()
        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_s = (step + 1) / elapsed
            print(
                f"  step {step + 1:4d}/{cfg.max_steps} | "
                f"loss={loss_val:.3f} | "
                f"lr={optim.param_groups[0]['lr']:.5f} | "
                f"{steps_per_s:.2f} step/s"
            )

        val_loss: float | None = None
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps:
            val_loss = evaluate(model, val_records, batch_size=cfg.batch_size)
            print(f"  >> val_loss={val_loss:.3f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.model.__dict__,
                    "step": step + 1,
                    "val_loss": val_loss,
                }, best_path)
                print(f"  >> saved checkpoint to {best_path}")

        if (step + 1) % cfg.sample_every == 0 or step + 1 == cfg.max_steps:
            print("  >> samples:")
            for ctx in sample_contexts:
                # context already ends with [SEP]; strip the trailing
                # marker so encode_context re-adds it.
                ctx_for_decode = ctx.removesuffix(" [SEP]")
                gen = sample_generation(
                    model, ctx_for_decode, max_new_tokens=80, temperature=0.0,
                )
                print(f"     {ctx_for_decode[:60]}…")
                print(f"     → {gen!r}")

        history.append((step + 1, loss_val, val_loss))

    elapsed = time.time() - t0
    print(
        f"Done. {cfg.max_steps} steps in {elapsed:.1f}s "
        f"({cfg.max_steps / elapsed:.2f} step/s) | "
        f"best val_loss={best_val:.3f}"
    )
    return {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "history": history,
        "elapsed_s": elapsed,
    }


def _infinite_batches(records, *, batch_size: int, seed: int):
    """Yield batches forever, reshuffling each epoch."""
    epoch = 0
    while True:
        for batch in iterate_batches(
            records, batch_size=batch_size, shuffle=True, seed=seed + epoch,
        ):
            yield batch
        epoch += 1
