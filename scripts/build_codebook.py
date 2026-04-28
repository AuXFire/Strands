#!/usr/bin/env python
"""Build the Semantic Strands codebook from seed concepts + WordNet.

Uses the layered build cache (see strands/build/cache.py) so repeated
runs only recompute layers whose inputs have changed. Edits to a single
seed file no longer trigger a full WordNet expansion.
"""

from __future__ import annotations

import argparse
import time
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
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the layer cache and rebuild everything from scratch.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the cache directory (default: strands/build/_cache).",
    )
    parser.add_argument(
        "--invalidate",
        nargs="*",
        choices=["wn", "morph", "sent", "all"],
        default=[],
        help="Layers to force-rebuild. 'all' clears every cached layer.",
    )
    parser.add_argument(
        "--drop-adjacency",
        action="store_true",
        help="Do not preserve any existing codon_adjacency in the output.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    codebook = write(
        args.output,
        frequency_threshold=args.frequency_threshold,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        invalidate=args.invalidate or None,
        preserve_adjacency=not args.drop_adjacency,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"Wrote {len(codebook['entries']):,} entries across "
        f"{len(codebook['domains'])} domains → {args.output}  ({elapsed:.1f}s total)"
    )


if __name__ == "__main__":
    main()
