#!/usr/bin/env python
"""Bake per-word ConceptNet relations into the codebook.

For every codebook entry, finds the two most-strongly-related codons
according to ConceptNet 5.7's raw assertion graph and writes them onto
the entry as a ``rel`` field. The encoder stamps these into each token's
strand-v2 binary representation, so compare-time relatedness scoring
becomes pure byte math on two strands — no codebook lookup, no runtime
model, no sidecar files.

Pipeline:
  1. Stream the gzipped assertions CSV (~475 MB / 34 M edges).
  2. Filter to English-English edges (/c/en/<word>) on both endpoints.
  3. For each edge, parse the JSON metadata for the per-edge weight and
     multiply by a per-relation scaling factor (Synonym ×1.20, RelatedTo
     ×1.00, IsA ×0.90, …; Antonym ×0.20, DistinctFrom ×0.10).
  4. Aggregate (word_a → other_codon) weighted scores via a defaultdict.
     Edges where the partner word maps to the same codon as the source
     are skipped (they're trivially same-concept matches in the
     comparator, not new information).
  5. For each word, take the top-2 codons by weighted score.
  6. Quantize per-word: weight_u8 = round(score / max_score × 255).

Inflectional variants inherit their base lemma's relations during the
codebook merge step, so a word like ``running`` carries the same
``rel`` as ``run``.
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
    "/r/Synonym":            1.20,
    "/r/SimilarTo":          1.10,
    "/r/RelatedTo":          1.00,
    "/r/IsA":                0.95,
    "/r/InstanceOf":         0.90,
    "/r/HasA":               0.85,
    "/r/PartOf":             0.85,
    "/r/MadeOf":             0.80,
    "/r/UsedFor":            0.90,
    "/r/CapableOf":          0.80,
    "/r/HasContext":         0.85,
    "/r/AtLocation":         0.75,
    "/r/HasProperty":        0.80,
    "/r/HasPrerequisite":    0.70,
    "/r/HasSubevent":        0.70,
    "/r/HasFirstSubevent":   0.65,
    "/r/HasLastSubevent":    0.65,
    "/r/Causes":             0.70,
    "/r/CausesDesire":       0.65,
    "/r/MotivatedByGoal":    0.70,
    "/r/DerivedFrom":        0.70,
    "/r/EtymologicallyRelatedTo": 0.55,
    "/r/MannerOf":           0.75,
    "/r/DefinedAs":          0.85,
    "/r/FormOf":             0.80,
    "/r/Entails":            0.75,
    "/r/HasSubclass":        0.85,
    "/r/Antonym":            0.25,
    "/r/DistinctFrom":       0.10,
}

_EN_PREFIX = "/c/en/"
_CONCEPT_RE = re.compile(r"^/c/en/([^/]+)")


def _extract_lemma(concept_uri: str) -> str | None:
    m = _CONCEPT_RE.match(concept_uri)
    if not m:
        return None
    return m.group(1).replace("_", " ")


def _codon_str(d: str, c: int, n: int) -> str:
    return f"{d}{c:01X}{n:02X}"


def build_word_to_codon(codebook: dict) -> dict[str, str]:
    return {
        word.lower(): _codon_str(raw["d"], raw["c"], raw["n"])
        for word, raw in codebook.get("entries", {}).items()
    }


def aggregate_word_relations(
    assertions_path: Path,
    word_to_codon: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Stream the assertions file and return ``word -> {codon: score}``."""
    word_relations: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    n_total = 0
    n_used = 0

    with gzip.open(assertions_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
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
            if codon_a is None or codon_b is None:
                continue
            try:
                meta = json.loads(meta_json)
                weight = float(meta.get("weight", 1.0))
            except (json.JSONDecodeError, ValueError):
                weight = 1.0
            score = weight * rel_w

            # Symmetrize: the relation contributes to both endpoints' graphs.
            # But skip self-codon edges (a -> codon_a) — comparator already
            # gives those a perfect concept match.
            if codon_b != codon_a:
                word_relations[lemma_a][codon_b] += score
            if codon_a != codon_b:
                word_relations[lemma_b][codon_a] += score

            n_used += 1
            if n_used % 250_000 == 0:
                print(f"  ... {n_used:,} edges aggregated (scanned {n_total:,})")

    print(f"  done: {n_used:,} edges used, {n_total:,} scanned")
    print(f"  {len(word_relations):,} words have at least one related codon")
    return word_relations


def _load_morph_cache(codebook: dict) -> dict | None:
    """Find the codebook's morphology cache file via its known location."""
    cache_dir = REPO_ROOT / "strands" / "build" / "_cache"
    if not cache_dir.is_dir():
        return None
    for path in cache_dir.glob("morph-*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            payload = blob.get("payload", {})
            if "edges" in payload:
                return payload
        except (OSError, json.JSONDecodeError):
            continue
    return None


def select_top_relations(
    word_relations: dict[str, dict[str, float]],
    top_k: int = 2,
) -> dict[str, list[list]]:
    """For each word, return its top-K related codons, with weights
    quantized to uint8 relative to the strongest neighbor."""
    out: dict[str, list[list]] = {}
    for word, codon_scores in word_relations.items():
        sorted_codons = sorted(codon_scores.items(), key=lambda x: -x[1])
        if not sorted_codons:
            continue
        max_score = sorted_codons[0][1]
        if max_score <= 0:
            continue
        top: list[list] = []
        for codon, score in sorted_codons[:top_k]:
            wq = int(round(score / max_score * 255))
            wq = max(1, min(255, wq))  # clamp to [1, 255] (0 is sentinel)
            top.append([codon, wq])
        out[word] = top
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--assertions", type=Path, default=DEFAULT_ASSERTIONS,
        help="Path to conceptnet-assertions-5.7.0.csv.gz",
    )
    parser.add_argument("--top-k", type=int, default=4)
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

    print(f"Building word→codon map …")
    w2c = build_word_to_codon(codebook)
    print(f"  {len(w2c):,} words")

    print(f"Streaming assertions from {args.assertions} …")
    word_relations = aggregate_word_relations(args.assertions, w2c)

    print(f"Selecting top-{args.top_k} relations per word …")
    top = select_top_relations(word_relations, top_k=args.top_k)
    print(f"  {len(top):,} words have stamped relations")

    # Stamp `rel` field onto each codebook entry.
    for word, raw in codebook["entries"].items():
        rels = top.get(word.lower())
        if rels:
            raw["rel"] = rels
        else:
            raw.pop("rel", None)

    # Inflectional propagation: a variant like "running" or "happier" that
    # has no ConceptNet edges of its own inherits its base lemma's
    # relations. We use the codebook's morphology cache to find variant→
    # base edges. Any variant whose base has `rel` gets a copy.
    print("Propagating relations from base lemmas to inflectional variants …")
    morph_layer = _load_morph_cache(codebook)
    inflection_propagated = 0
    if morph_layer is not None:
        for variant, base, _pos in morph_layer.get("edges", []):
            v_raw = codebook["entries"].get(variant)
            if v_raw is None or "rel" in v_raw:
                continue
            base_raw = codebook["entries"].get(base)
            if base_raw is None or "rel" not in base_raw:
                continue
            # Deep-copy the rel list so independent edits don't alias.
            v_raw["rel"] = [list(r) for r in base_raw["rel"]]
            inflection_propagated += 1
    print(f"  propagated relations to {inflection_propagated:,} inflected variants")

    # Drop any leftover legacy fields from previous experiments.
    codebook.pop("codon_adjacency", None)
    codebook.pop("embedded_vectors", None)
    if "stats" in codebook:
        codebook["stats"].pop("codon_adjacency_edges", None)
        codebook["stats"].pop("codon_adjacency_source", None)
    for word, raw in codebook["entries"].items():
        raw.pop("vec_idx", None)

    print(f"Writing codebook to {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    size = args.output.stat().st_size
    print(f"  total codebook size: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
