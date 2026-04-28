#!/usr/bin/env python
"""Wrapper script: build the codebook JSON from seed concepts + WordNet."""

from __future__ import annotations

import argparse
from pathlib import Path

from strands.build.assemble import write


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Semantic Strands codebook.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "strands" / "data" / "codebook_v0.1.0.json",
    )
    parser.add_argument(
        "--frequency-threshold",
        type=float,
        default=None,
        help="Drop non-seed entries below this Zipf frequency. Default: full vocab.",
    )
    args = parser.parse_args()

    codebook = write(args.output, frequency_threshold=args.frequency_threshold)
    print(
        f"Wrote {len(codebook['entries'])} entries across "
        f"{len(codebook['domains'])} domains → {args.output}"
    )


if __name__ == "__main__":
    main()
