#!/usr/bin/env python
"""Build a codon→codon adjacency table from ConceptNet 5.7 raw assertions.

This is the strand-native, embedding-free relatedness graph. It replaces
the Numberbatch-derived adjacency with one built directly from
ConceptNet's typed edges, preserving relation type weights and source
attribution.

Pipeline:
  1. Stream the gzipped assertions CSV (~475 MB compressed, ~34 M edges).
  2. Filter to English-English edges (/c/en/<word> on both sides).
  3. Parse the per-edge weight + relation type from the JSON metadata.
  4. Map each lemma to its codon via the existing codebook.
  5. For each (codon_a, codon_b, relation), accumulate the sum of edge
     weights × per-relation factor.
  6. For each codon, retain top-N neighbors (default 32). Quantize the
     accumulated weight to uint8 by dividing by per-codon max.
  7. Serialize as adjacency in codebook JSON.

Per-relation weighting reflects how informative each relation is for
similarity / relatedness benchmarks. Synonym edges are strongest;
RelatedTo and IsA are good signals; HasContext / AtLocation give
weaker but useful relatedness; pure noise relations (ExternalURL,
dbpedia attribution) are dropped entirely.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "strands" / "data" / "codebook_v0.1.0.json"
DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_ASSERTIONS = Path("/tmp/conceptnet/assertions.csv.gz")

# Per-relation multipliers — higher = stronger evidence of relatedness.
# Relations not in this dict are dropped entirely.
RELATION_WEIGHTS: dict[str, float] = {
    "/r/Synonym":           1.20,
    "/r/RelatedTo":         1.00,
    "/r/IsA":               0.90,
    "/r/PartOf":            0.85,
    "/r/MadeOf":            0.80,
    "/r/HasA":              0.85,
    "/r/UsedFor":           0.90,
    "/r/CapableOf":         0.80,
    "/r/AtLocation":        0.75,
    "/r/HasContext":        0.85,
    "/r/HasProperty":       0.80,
    "/r/Causes":            0.70,
    "/r/CausesDesire":      0.65,
    "/r/MotivatedByGoal":   0.70,
    "/r/HasPrerequisite":   0.70,
    "/r/HasSubevent":       0.70,
    "/r/HasFirstSubevent":  0.65,
    "/r/HasLastSubevent":   0.65,
    "/r/SimilarTo":         1.10,
    "/r/DerivedFrom":       0.70,
    "/r/EtymologicallyRelatedTo": 0.65,
    "/r/MannerOf":          0.75,
    "/r/DefinedAs":         0.85,
    "/r/InstanceOf":        0.85,
    "/r/FormOf":            0.80,
    "/r/Entails":           0.75,
    "/r/HasSubclass":       0.85,
    # Antonyms are deliberately weighted low — they're related but
    # opposite, which we don't want as a strong relatedness signal.
    "/r/Antonym":           0.20,
    "/r/DistinctFrom":      0.10,
}


_EN_PREFIX = "/c/en/"
_CONCEPT_RE = re.compile(r"^/c/en/([^/]+)")


def _extract_lemma(concept_uri: str) -> str | None:
    """Return the bare English lemma from a /c/en/<word>[/pos] URI."""
    m = _CONCEPT_RE.match(concept_uri)
    if not m:
        return None
    return m.group(1).replace("_", " ")


def _codon_str(d: str, c: int, n: int) -> str:
    return f"{d}{c:01X}{n:02X}"


def _build_word_to_codon(codebook: dict) -> dict[str, str]:
    """Word → primary codon string."""
    out: dict[str, str] = {}
    for word, raw in codebook.get("entries", {}).items():
        out[word.lower()] = _codon_str(raw["d"], raw["c"], raw["n"])
    return out


def stream_edges(
    assertions_path: Path,
    word_to_codon: dict[str, str],
    *,
    max_edges: int | None = None,
):
    """Yield ``(codon_a, codon_b, weighted_score)`` for English edges
    where both endpoints are in the codebook."""
    n_total = 0
    n_emitted = 0

    with gzip.open(assertions_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            if max_edges and n_emitted >= max_edges:
                break
            try:
                _, rel, src, tgt, meta_json = line.rstrip("\n").split("\t", 4)
            except ValueError:
                continue
            if not (src.startswith(_EN_PREFIX) and tgt.startswith(_EN_PREFIX)):
                continue
            rel_w = RELATION_WEIGHTS.get(rel)
            if rel_w is None:
                continue
            lemma_a = _extract_lemma(src)
            lemma_b = _extract_lemma(tgt)
            if not lemma_a or not lemma_b or lemma_a == lemma_b:
                continue
            codon_a = word_to_codon.get(lemma_a)
            codon_b = word_to_codon.get(lemma_b)
            if codon_a is None or codon_b is None or codon_a == codon_b:
                continue
            try:
                meta = json.loads(meta_json)
                weight = float(meta.get("weight", 1.0))
            except (json.JSONDecodeError, ValueError):
                weight = 1.0
            score = weight * rel_w
            yield codon_a, codon_b, score
            n_emitted += 1
            if n_emitted % 250_000 == 0:
                print(f"  ... {n_emitted:,} edges emitted (scanned {n_total:,})")
    print(f"  done: {n_emitted:,} edges emitted, {n_total:,} total scanned")


def build_adjacency(
    codebook: dict,
    assertions_path: Path,
    *,
    neighbors_per_codon: int = 32,
    min_relative_weight: float = 0.05,
) -> dict[str, list[list]]:
    print("Building word → codon map …")
    w2c = _build_word_to_codon(codebook)
    print(f"  {len(w2c):,} words mapped to codons")

    print(f"Streaming ConceptNet edges from {assertions_path} …")
    pair_weights: dict[tuple[str, str], float] = defaultdict(float)
    for codon_a, codon_b, score in stream_edges(assertions_path, w2c):
        # Symmetrize: we want symmetric relatedness.
        if codon_a < codon_b:
            pair_weights[(codon_a, codon_b)] += score
        else:
            pair_weights[(codon_b, codon_a)] += score

    print(f"  aggregated {len(pair_weights):,} unique codon pairs")

    # Build per-codon outgoing adjacency, normalized + quantized.
    out_edges: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), w in pair_weights.items():
        out_edges[a].append((b, w))
        out_edges[b].append((a, w))

    adjacency: dict[str, list[list]] = {}
    for codon, edges in out_edges.items():
        edges.sort(key=lambda e: -e[1])
        if not edges:
            continue
        max_w = edges[0][1]
        kept: list[list] = []
        for nb, w in edges[:neighbors_per_codon]:
            rel = w / max_w if max_w > 0 else 0
            if rel < min_relative_weight:
                break
            wq = int(round(rel * 255))
            kept.append([nb, wq])
        if kept:
            adjacency[codon] = kept

    total_edges = sum(len(v) for v in adjacency.values())
    print(f"  {len(adjacency):,} codons with edges; {total_edges:,} edges total")
    return adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--assertions", type=Path, default=DEFAULT_ASSERTIONS,
        help="Path to conceptnet-assertions-5.7.0.csv.gz",
    )
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument(
        "--min-relative-weight", type=float, default=0.05,
        help="Drop edges weighted below this fraction of each codon's strongest edge.",
    )
    args = parser.parse_args()

    if not args.assertions.exists():
        raise SystemExit(
            f"Assertions file not found: {args.assertions}\n"
            f"Download from "
            f"https://conceptnet.s3.amazonaws.com/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
        )

    print(f"Loading codebook from {args.codebook}")
    with args.codebook.open("r", encoding="utf-8") as f:
        codebook = json.load(f)

    adjacency = build_adjacency(
        codebook, args.assertions,
        neighbors_per_codon=args.neighbors,
        min_relative_weight=args.min_relative_weight,
    )

    codebook["codon_adjacency"] = adjacency
    codebook.setdefault("stats", {})["codon_adjacency_edges"] = sum(
        len(v) for v in adjacency.values()
    )
    codebook["stats"]["codon_adjacency_source"] = "conceptnet-5.7.0"

    print(f"Writing codebook with adjacency to {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    size = args.output.stat().st_size
    print(f"  total codebook size: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
