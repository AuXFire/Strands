"""Drive the deterministic system over a generated corpus and
capture (Conditioning, deterministic_answer) pairs to JSONL for
neural Compute Module training.

Usage:
    python scripts/capture_training_data.py \
        --output strands/data/compute/train.jsonl \
        --max-per-category 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from strands.backbone import (
    DiscourseState,
    RecordingComputeModule,
    StubComputeModule,
    load,
    respond,
)
from strands.compute.corpus import (
    generate_multiturn_sessions,
    generate_prompts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("strands/data/compute/train.jsonl"),
    )
    parser.add_argument(
        "--max-per-category", type=int, default=None,
        help="Cap prompts per category. None = no cap.",
    )
    parser.add_argument(
        "--multiturn-sessions", type=int, default=20,
    )
    parser.add_argument(
        "--confidence-floor", type=float, default=0.99,
        help="Force every turn through the seam (default) or set to "
             "0.5 to only capture low-confidence cases.",
    )
    parser.add_argument(
        "--backbone-dir", type=Path,
        default=Path("strands/data/backbone"),
    )
    args = parser.parse_args()

    print(f"Loading backbone from {args.backbone_dir} …")
    backbone = load(args.backbone_dir)
    print(
        f"  {len(backbone.nodes)} nodes, "
        f"{len(backbone.edges)} edges"
    )

    print("Generating single-turn corpus …")
    prompts = generate_prompts(
        backbone, max_per_category=args.max_per_category,
    )
    print(f"  {len(prompts)} single-turn prompts")

    print(f"Generating {args.multiturn_sessions} multi-turn sessions …")
    sessions = generate_multiturn_sessions(
        backbone, n_sessions=args.multiturn_sessions,
    )
    n_multi = sum(len(s) for s in sessions)
    print(f"  {n_multi} multi-turn prompts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
        print(f"  cleared existing {args.output}")

    inner = StubComputeModule(override=None)  # defer to deterministic
    rec = RecordingComputeModule(args.output, inner=inner)
    t0 = time.time()
    captured = 0

    try:
        # Single-turn: fresh state per prompt.
        for i, p in enumerate(prompts):
            state = DiscourseState()
            respond(
                backbone, p.text, state=state, compute=rec,
                confidence_floor=args.confidence_floor,
            )
            captured += 1
            if (i + 1) % 200 == 0:
                print(
                    f"  single-turn: {i + 1}/{len(prompts)} "
                    f"({captured / (time.time() - t0):.1f}/s)"
                )

        # Multi-turn: shared state across the session.
        for s_idx, session in enumerate(sessions):
            state = DiscourseState()
            for p in session:
                respond(
                    backbone, p.text, state=state, compute=rec,
                    confidence_floor=args.confidence_floor,
                )
                captured += 1
            if (s_idx + 1) % 5 == 0:
                print(f"  multi-turn: session {s_idx + 1}/{len(sessions)}")
    finally:
        rec.close()

    elapsed = time.time() - t0
    print(
        f"Done. Captured {captured} examples in {elapsed:.1f}s "
        f"({captured / elapsed:.1f}/s) → {args.output}"
    )
    size_kb = args.output.stat().st_size / 1024
    print(f"  output size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
