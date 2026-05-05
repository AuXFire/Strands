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
    # Hardware. 'auto' picks cuda when available (which on a ROCm
    # PyTorch build presents the AMD GPU as cuda), else cpu.
    device: str = "auto"
    # Early stopping: stop training if val loss hasn't improved in
    # ``patience`` consecutive evaluations. 0 disables.
    patience: int = 5


def resolve_device(spec: str) -> torch.device:
    """Resolve 'auto' to cuda-when-available else cpu. ROCm builds of
    PyTorch present AMD GPUs through the cuda namespace, so the same
    string works for both NVIDIA and AMD."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def describe_device(device: torch.device) -> str:
    """One-line description of the device for logging."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_gb = (
            torch.cuda.get_device_properties(device).total_memory / 1e9
        )
        # ROCm builds report the AMD card here; the API is identical.
        backend = (
            "ROCm" if getattr(torch.version, "hip", None) is not None
            else "CUDA"
        )
        return f"{name} ({backend}, {total_gb:.1f} GB)"
    threads = torch.get_num_threads()
    return f"CPU ({threads} threads)"


def lr_at_step(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup → cosine decay to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def compute_loss(
    model: TinyTransformer, batch: dict[str, torch.Tensor],
    *, device: torch.device | None = None,
) -> torch.Tensor:
    """Standard causal-LM loss masked to target tokens only.
    ``device`` (if supplied) moves the batch tensors before the
    forward pass."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if device is not None:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
    batch_size: int, device: torch.device | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in iterate_batches(val_records, batch_size=batch_size, shuffle=False):
        loss = compute_loss(model, batch, device=device)
        total_loss += float(loss)
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


@torch.no_grad()
def sample_generation(
    model: TinyTransformer, context: str, *,
    max_new_tokens: int = 96, temperature: float = 0.0,
    device: torch.device | None = None,
) -> str:
    """Greedy decode from a context string. Returns the generated
    target portion (everything after [SEP])."""
    model.eval()
    ids = torch.tensor([encode_context(context)], dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, eos_id=EOS_ID,
        temperature=temperature,
    )
    # Strip the context portion.
    generated_ids = out[0, ids.shape[1]:].tolist()
    return decode(generated_ids)


def train(cfg: TrainingConfig) -> dict:
    """Run training loop. Returns a dict with final metrics + best
    checkpoint path. Honors cfg.device (cuda/rocm-as-cuda/cpu) and
    cfg.patience (early stopping)."""
    device = resolve_device(cfg.device)
    print(f"Device: {describe_device(device)}")

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
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = TinyTransformer(cfg.model).to(device)
    print(f"  {model.num_parameters():,} parameters")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = cfg.out_dir / "best.pt"
    evals_since_improvement = 0
    early_stopped = False

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
        loss = compute_loss(model, batch, device=device)
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
            val_loss = evaluate(
                model, val_records, batch_size=cfg.batch_size, device=device,
            )
            print(f"  >> val_loss={val_loss:.3f}")
            if val_loss < best_val:
                best_val = val_loss
                evals_since_improvement = 0
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.model.__dict__,
                    "step": step + 1,
                    "val_loss": val_loss,
                }, best_path)
                print(f"  >> saved checkpoint to {best_path}")
            else:
                evals_since_improvement += 1
                if cfg.patience > 0 and evals_since_improvement >= cfg.patience:
                    print(
                        f"  >> early stop: no val improvement for "
                        f"{cfg.patience} evals (best={best_val:.3f})"
                    )
                    early_stopped = True

        if (step + 1) % cfg.sample_every == 0 or step + 1 == cfg.max_steps:
            print("  >> samples:")
            for ctx in sample_contexts:
                ctx_for_decode = ctx.removesuffix(" [SEP]")
                gen = sample_generation(
                    model, ctx_for_decode, max_new_tokens=80,
                    temperature=0.0, device=device,
                )
                print(f"     {ctx_for_decode[:60]}…")
                print(f"     → {gen!r}")

        history.append((step + 1, loss_val, val_loss))
        if early_stopped:
            break

    elapsed = time.time() - t0
    actual_steps = len(history)
    print(
        f"Done. {actual_steps} steps in {elapsed:.1f}s "
        f"({actual_steps / max(elapsed, 1e-9):.2f} step/s) | "
        f"best val_loss={best_val:.3f}"
    )
    return {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "history": history,
        "elapsed_s": elapsed,
        "early_stopped": early_stopped,
        "actual_steps": actual_steps,
        "device": str(device),
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
