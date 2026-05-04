"""Evaluate the trained neural Compute Module against the training
set: how often does it produce a byte-exact match? How often does
it produce something that overlaps with the deterministic answer?

This isn't a generalization metric — it's a fit metric. A high
exact-match rate means the model has memorized the deterministic
mapping (good for the test training loop). A low rate means the
model is undertrained.

Usage:
    python scripts/eval_neural_module.py [--n 50] [--temperature 0.0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from strands.backbone import dict_to_conditioning
from strands.compute import (
    NeuralComputeModule,
    decode,
    encode_context,
)
from strands.compute.format import format_context


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("strands/data/compute/checkpoints/best.pt"))
    p.add_argument("--data", type=Path,
                   default=Path("strands/data/compute/train.jsonl"))
    p.add_argument("--n", type=int, default=50,
                   help="Sample N records from the training set.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cm = NeuralComputeModule.from_checkpoint(
        args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"Loaded checkpoint: {cm.model.cfg}")
    print(f"  params: {cm.model.num_parameters():,}")

    lines = args.data.read_text(encoding="utf-8").splitlines()
    import random
    rng = random.Random(args.seed)
    sample = rng.sample(lines, min(args.n, len(lines)))

    exact_matches = 0
    prefix_matches = 0  # generated answer starts with the same first 8 chars
    nonempty = 0
    total_time = 0.0
    examples: list[tuple[str, str, str]] = []

    for line in sample:
        d = json.loads(line)
        cond = dict_to_conditioning(d["conditioning"])
        target = d["deterministic_answer"].strip()

        ctx = format_context(cond)
        ids = torch.tensor([encode_context(ctx)], dtype=torch.long)
        if ids.shape[1] >= cm.model.cfg.max_seq_len:
            continue

        t0 = time.time()
        with torch.no_grad():
            out = cm.model.generate(
                ids,
                max_new_tokens=min(
                    cm.max_new_tokens,
                    cm.model.cfg.max_seq_len - ids.shape[1],
                ),
                eos_id=258,
                temperature=cm.temperature,
                top_k=cm.top_k,
            )
        total_time += time.time() - t0
        gen = decode(out[0, ids.shape[1]:].tolist()).strip()

        if gen:
            nonempty += 1
        if gen == target:
            exact_matches += 1
        if gen and target and gen[:8] == target[:8]:
            prefix_matches += 1

        if len(examples) < 10:
            examples.append((cond.prompt, target, gen))

    n = len(sample)
    print()
    print(f"=== Eval over {n} training examples ===")
    print(f"  exact matches:   {exact_matches}/{n} ({100*exact_matches/n:.1f}%)")
    print(f"  prefix matches:  {prefix_matches}/{n} ({100*prefix_matches/n:.1f}%)")
    print(f"  non-empty:       {nonempty}/{n} ({100*nonempty/n:.1f}%)")
    print(f"  avg gen time:    {total_time/n*1000:.0f}ms")
    print()
    print("Sample comparisons:")
    for prompt, target, gen in examples:
        print()
        print(f"  > {prompt}")
        print(f"    target: {target}")
        print(f"    neural: {gen}")


if __name__ == "__main__":
    main()
