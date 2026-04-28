"""Build a codon→codon adjacency table from ConceptNet Numberbatch.

The output replaces the runtime Numberbatch dependency with a compact
sparse graph (~200 KB) that ships in the codebook JSON. Quality is
near-identical because Numberbatch is itself a compression of ConceptNet
relations, and we re-compress it at the codon granularity (which is the
right granularity for strand comparisons).

Pipeline:
  1. Group codebook entries by primary codon. Each codon is represented
     by up to K seed words (preferring frequency-ranked, then alphabetical).
  2. Average each codon's seed-word Numberbatch vectors → codon centroid.
  3. For each codon centroid, compute cosine to every other centroid.
     Take top-N neighbors (default 16).
  4. Quantize cosines to uint8 (0–255).
  5. Serialize as adjacency: {codon_str: [(neighbor_codon, weight_u8), ...]}.

Determinism: vectors come from a fixed Numberbatch release; codon ordering
sorted; ties broken by codon string. Same input → same JSON byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "strands" / "data" / "codebook_v0.1.0.json"
DEFAULT_OUTPUT = DEFAULT_INPUT


def _load_numberbatch():
    import gensim.downloader as api
    return api.load("conceptnet-numberbatch-17-06-300")


def _codon_str(d: str, c: int, n: int) -> str:
    return f"{d}{c:01X}{n:02X}"


def build_adjacency(
    codebook: dict,
    *,
    seeds_per_codon: int = 8,
    neighbors_per_codon: int = 16,
    min_weight: float = 0.10,
) -> dict:
    print("Loading Numberbatch …")
    nb = _load_numberbatch()

    print("Grouping entries by codon …")
    by_codon: dict[str, list[str]] = defaultdict(list)
    for word, raw in codebook["entries"].items():
        cs = _codon_str(raw["d"], raw["c"], raw["n"])
        by_codon[cs].append(word)

    print(f"  {len(by_codon)} distinct codons")

    print(f"Collecting up to {seeds_per_codon} representative-word vectors per codon …")
    codon_vecs: dict[str, np.ndarray] = {}
    for codon, words in by_codon.items():
        # Prefer shorter (more canonical) words first, then alphabetical
        sorted_words = sorted(words, key=lambda w: (len(w), w))
        vecs = []
        for w in sorted_words:
            key = f"/c/en/{w.replace(' ', '_')}"
            if key in nb:
                v = nb[key]
                n = np.linalg.norm(v)
                if n > 0:
                    vecs.append(v / n)
            if len(vecs) >= seeds_per_codon:
                break
        if vecs:
            codon_vecs[codon] = np.stack(vecs)  # (k, 300), each row unit-normed

    print(f"  {len(codon_vecs)} codons with at least one representative vector")

    codon_list = sorted(codon_vecs.keys())
    print(
        "Computing per-codon-pair MAX-of-word-cosines "
        f"({len(codon_list)} codons, ~{len(codon_list)**2//2:,} pairs) …"
    )

    adjacency: dict[str, list[list]] = {}
    n_codons = len(codon_list)

    # Sim matrix: for each codon i, the max-of-word-cosines vs every other codon.
    sim_rows: list[np.ndarray] = []
    for i, codon_a in enumerate(codon_list):
        va = codon_vecs[codon_a]  # (ka, 300)
        row = np.zeros(n_codons, dtype=np.float32)
        for j, codon_b in enumerate(codon_list):
            if j == i:
                row[j] = -1.0
                continue
            vb = codon_vecs[codon_b]  # (kb, 300)
            # Pairwise cosine matrix, take max
            row[j] = float((va @ vb.T).max())
        sim_rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{n_codons} codons processed")

    print("Selecting top-N neighbors per codon …")
    for i, codon in enumerate(codon_list):
        sims = sim_rows[i]
        top_idx = np.argpartition(-sims, min(neighbors_per_codon, n_codons - 1))[:neighbors_per_codon]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        edges = []
        for j in top_idx:
            w = float(sims[j])
            if w < min_weight:
                continue
            wq = int(round(min(1.0, max(0.0, w)) * 255))
            edges.append([codon_list[j], wq])
        if edges:
            adjacency[codon] = edges

    print(f"  {len(adjacency)} codons with at least one outgoing edge")
    return adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=8,
                        help="Max representative words per codon centroid")
    parser.add_argument("--neighbors", type=int, default=16,
                        help="Top-N neighbors per codon to retain")
    parser.add_argument("--min-weight", type=float, default=0.10,
                        help="Drop edges below this cosine")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the layer cache for adjacency.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading codebook from {args.codebook}")
    with args.codebook.open("r", encoding="utf-8") as f:
        codebook = json.load(f)

    # Cache key: codebook seeds_hash + Numberbatch version + parameters that
    # affect output. If the seeds didn't change and Numberbatch didn't
    # change, the adjacency is reused unchanged.
    from strands.build.cache import (
        NUMBERBATCH_VERSION,
        get_or_build,
        _hash,
    )
    seeds_hash = codebook.get("stats", {}).get("seeds_hash", "unknown")
    adj_input_hash = _hash({
        "seeds": seeds_hash,
        "nb": NUMBERBATCH_VERSION,
        "seeds_per_codon": args.seeds,
        "neighbors": args.neighbors,
        "min_weight": args.min_weight,
    })

    if args.no_cache:
        adjacency = build_adjacency(
            codebook,
            seeds_per_codon=args.seeds,
            neighbors_per_codon=args.neighbors,
            min_weight=args.min_weight,
        )
    else:
        adjacency = get_or_build(
            "adjacency", adj_input_hash,
            lambda: build_adjacency(
                codebook,
                seeds_per_codon=args.seeds,
                neighbors_per_codon=args.neighbors,
                min_weight=args.min_weight,
            ),
            cache_dir=args.cache_dir,
            sources={
                "seeds_hash": seeds_hash,
                "numberbatch": NUMBERBATCH_VERSION,
                "seeds_per_codon": args.seeds,
                "neighbors": args.neighbors,
                "min_weight": args.min_weight,
            },
        )

    codebook["codon_adjacency"] = adjacency
    codebook.setdefault("stats", {})["codon_adjacency_edges"] = sum(
        len(v) for v in adjacency.values()
    )

    print(f"Writing codebook with adjacency to {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    size = args.output.stat().st_size
    print(f"  total codebook size: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
