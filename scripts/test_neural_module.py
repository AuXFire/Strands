"""Smoke-test a trained neural Compute Module checkpoint.

Loads the checkpoint, swaps it in for the deterministic system on a
suite of prompts, and prints (deterministic_answer, neural_answer)
side by side so you can see what the NN is doing.

Usage:
    python scripts/test_neural_module.py \
        [--checkpoint strands/data/compute/checkpoints/best.pt] \
        [--temperature 0.0]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from strands.backbone import DiscourseState, load, respond
from strands.compute import NeuralComputeModule


_PROMPTS = [
    "What is a cat?",
    "What is a dog?",
    "What is a bird?",
    "Where do birds live?",
    "What can a fish do?",
    "Is a cat an animal?",
    "Is a cat a dog?",
    "Can a bird fly?",
    "Hello there",
    "Thanks for your help",
    "Tell me more.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("strands/data/compute/checkpoints/best.pt"))
    p.add_argument("--backbone-dir", type=Path,
                   default=Path("strands/data/backbone"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=120)
    args = p.parse_args()

    print(f"Loading backbone from {args.backbone_dir} …")
    backbone = load(args.backbone_dir)

    print(f"Loading checkpoint from {args.checkpoint} …")
    cm = NeuralComputeModule.from_checkpoint(
        args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"  model: {cm.model.cfg}")
    print(f"  params: {cm.model.num_parameters():,}")
    print()

    for prompt in _PROMPTS:
        # Force the seam to fire so we always see what the NN produces.
        state_det = DiscourseState()
        state_neural = DiscourseState()
        det = respond(backbone, prompt, state=state_det)
        t0 = time.time()
        neu = respond(
            backbone, prompt, state=state_neural, compute=cm,
            confidence_floor=0.99,
        )
        elapsed = time.time() - t0

        print(f">>> {prompt}")
        print(f"  deterministic ({det.confidence:.2f}): {det.text}")
        print(
            f"  neural        ({elapsed:.2f}s, "
            f"{'override' if neu.compute_module_used else 'defer'}): "
            f"{neu.text}"
        )
        print()


if __name__ == "__main__":
    main()
