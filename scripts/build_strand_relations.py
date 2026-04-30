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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "strands" / "data" / "codebook_v0.1.0.json"
DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_ASSERTIONS = Path("/tmp/conceptnet/assertions.csv.gz")
DEFAULT_NUMBERBATCH = Path("/tmp/conceptnet/numberbatch-en-19.08.txt.gz")
DEFAULT_SWOW = Path("/tmp/swow/expanded")

# Per-relation multipliers — higher = stronger evidence of relatedness.
# Relations not in this dict are dropped entirely.
RELATION_POLICY: dict[str, tuple[str, float]] = {
    "/r/Synonym":            ("SYN", 1.20),
    "/r/SimilarTo":          ("SYN", 1.10),
    "/r/RelatedTo":          ("RELATED", 1.00),
    "/r/IsA":                ("ISA", 0.95),
    "/r/InstanceOf":         ("ISA", 0.90),
    "/r/HasA":               ("PART", 0.85),
    "/r/PartOf":             ("PART", 0.85),
    "/r/MadeOf":             ("PART", 0.80),
    "/r/UsedFor":            ("USED_FOR", 0.90),
    "/r/CapableOf":          ("USED_FOR", 0.80),
    "/r/HasContext":         ("CTX", 0.85),
    "/r/AtLocation":         ("CTX", 0.75),
    "/r/HasProperty":        ("PROP", 0.80),
    "/r/HasPrerequisite":    ("CAUSES", 0.70),
    "/r/HasSubevent":        ("ENTAILS", 0.70),
    "/r/HasFirstSubevent":   ("ENTAILS", 0.65),
    "/r/HasLastSubevent":    ("ENTAILS", 0.65),
    "/r/Causes":             ("CAUSES", 0.70),
    "/r/CausesDesire":       ("CAUSES", 0.65),
    "/r/MotivatedByGoal":    ("CAUSES", 0.70),
    "/r/DerivedFrom":        ("RELATED", 0.70),
    "/r/EtymologicallyRelatedTo": ("RELATED", 0.55),
    "/r/MannerOf":           ("ISA", 0.75),
    "/r/DefinedAs":          ("SYN", 0.85),
    "/r/FormOf":             ("SYN", 0.80),
    "/r/Entails":            ("ENTAILS", 0.75),
    "/r/HasSubclass":        ("ISA", 0.85),
    "/r/Antonym":            ("ANTI", 1.00),
    "/r/DistinctFrom":       ("ANTI", 0.80),
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
) -> dict[str, dict[tuple[str, str], float]]:
    """Stream assertions and return ``word -> {(type, codon): score}``."""
    word_relations: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
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
            policy = RELATION_POLICY.get(rel)
            if policy is None:
                continue
            rel_type, rel_w = policy
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
                word_relations[lemma_a][(rel_type, codon_b)] += score
            if codon_a != codon_b:
                word_relations[lemma_b][(rel_type, codon_a)] += score

            n_used += 1
            if n_used % 250_000 == 0:
                print(f"  ... {n_used:,} edges aggregated (scanned {n_total:,})")

    print(f"  done: {n_used:,} edges used, {n_total:,} scanned")
    print(f"  {len(word_relations):,} words have at least one related codon")
    return word_relations


def _numberbatch_token_to_word(token: str) -> str:
    if token.startswith("/c/en/"):
        token = token[len("/c/en/"):]
    return token.replace("_", " ").lower()


def _normalize_free_assoc_word(word: str) -> str:
    return word.strip().replace("_", " ").lower()


def _read_numberbatch_subset(
    numberbatch_path: Path,
    word_to_codon: dict[str, str],
) -> tuple[list[str], "np.ndarray"]:
    import numpy as np

    words: list[str] = []
    vectors: list[np.ndarray] = []
    wanted = set(word_to_codon)
    with gzip.open(numberbatch_path, "rt", encoding="utf-8") as f:
        header = next(f, "")
        dims = int(header.split()[1]) if len(header.split()) == 2 else 300
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dims + 1:
                continue
            word = _numberbatch_token_to_word(parts[0])
            if word not in wanted:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                continue
            words.append(word)
            vectors.append(vec / norm)
            if len(words) % 50_000 == 0:
                print(f"  ... loaded {len(words):,} matching Numberbatch vectors")
    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    return words, np.vstack(vectors).astype(np.float32, copy=False)


def augment_with_numberbatch(
    word_relations: dict[str, dict[tuple[str, str], float]],
    numberbatch_path: Path,
    word_to_codon: dict[str, str],
    *,
    top_k: int,
    min_cosine: float,
    scale: float,
    fill_only: bool,
    batch_size: int = 4096,
) -> int:
    """Distill Numberbatch 19.08 into native RELATED codon edges."""
    import numpy as np

    words, vectors = _read_numberbatch_subset(numberbatch_path, word_to_codon)
    if len(words) == 0:
        return 0

    codon_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, word in enumerate(words):
        codon_to_indices[word_to_codon[word]].append(idx)

    codons: list[str] = []
    prototypes: list[np.ndarray] = []
    for codon, indices in codon_to_indices.items():
        proto = vectors[indices].mean(axis=0)
        norm = float(np.linalg.norm(proto))
        if norm <= 0:
            continue
        codons.append(codon)
        prototypes.append(proto / norm)
    if not prototypes:
        return 0

    proto_matrix = np.vstack(prototypes).astype(np.float32, copy=False)
    added = 0
    for start in range(0, len(words), batch_size):
        stop = min(start + batch_size, len(words))
        sims = vectors[start:stop] @ proto_matrix.T
        for row_idx, word in enumerate(words[start:stop]):
            if fill_only and word_relations.get(word):
                continue
            own_codon = word_to_codon[word]
            row = sims[row_idx]
            if top_k < len(row):
                candidate_idx = np.argpartition(row, -top_k - 1)[-(top_k + 1):]
                candidate_idx = candidate_idx[np.argsort(row[candidate_idx])[::-1]]
            else:
                candidate_idx = np.argsort(row)[::-1]
            kept = 0
            for idx in candidate_idx:
                codon = codons[int(idx)]
                if codon == own_codon:
                    continue
                cosine = float(row[int(idx)])
                if cosine < min_cosine:
                    continue
                word_relations[word][("TOPIC", codon)] += cosine * scale
                added += 1
                kept += 1
                if kept >= top_k:
                    break
        if stop % 50_000 == 0 or stop == len(words):
            print(f"  ... distilled Numberbatch for {stop:,}/{len(words):,} words")
    return added


def augment_with_swow(
    word_relations: dict[str, dict[tuple[str, str], float]],
    swow_dir: Path,
    word_to_codon: dict[str, str],
    *,
    relation_type: str,
    top_k: int,
    min_weight: float,
    scale: float,
    reverse_scale: float,
    fill_only: bool,
) -> int:
    """Distill SWOW free associations into native typed codon edges."""
    names_path = swow_dir / "names.tsv"
    adjacency_path = swow_dir / "adjacency.tsv"
    if not names_path.exists() or not adjacency_path.exists():
        raise SystemExit(
            f"SWOW files not found in {swow_dir}; expected names.tsv and adjacency.tsv"
        )

    names = [
        _normalize_free_assoc_word(line)
        for line in names_path.read_text(encoding="utf-8").splitlines()
    ]
    assoc_by_word: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    scanned = 0
    kept = 0

    with adjacency_path.open("r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                src_raw, dst_raw, weight_raw = line.rstrip("\n").split("\t")
                src_idx = int(src_raw)
                dst_idx = int(dst_raw)
                weight = float(weight_raw)
            except ValueError:
                continue
            if weight < min_weight:
                continue
            if src_idx < 0 or dst_idx < 0 or src_idx >= len(names) or dst_idx >= len(names):
                continue
            src = names[src_idx]
            dst = names[dst_idx]
            if not src or not dst or src == dst:
                continue
            codon_src = word_to_codon.get(src)
            codon_dst = word_to_codon.get(dst)
            if codon_src is None or codon_dst is None or codon_src == codon_dst:
                continue

            score = weight * scale
            assoc_by_word[src][codon_dst] += score
            if reverse_scale > 0:
                assoc_by_word[dst][codon_src] += score * reverse_scale
            kept += 1

    added = 0
    for word, codon_scores in assoc_by_word.items():
        if fill_only and word_relations.get(word):
            continue
        for codon, score in sorted(codon_scores.items(), key=lambda x: -x[1])[:top_k]:
            word_relations[word][(relation_type, codon)] += score
            added += 1

    print(f"  scanned {scanned:,} SWOW edges; kept {kept:,} codebook-covered edges")
    print(f"  added {added:,} SWOW-derived {relation_type} candidates")
    return added


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
    word_relations: dict[str, dict[tuple[str, str], float]],
    top_k: int = 2,
    *,
    balanced_lanes: bool = False,
) -> dict[str, list[list]]:
    """For each word, return its top-K related codons, with weights
    quantized to uint8 relative to the strongest neighbor."""
    out: dict[str, list[list]] = {}
    for word, typed_scores in word_relations.items():
        sorted_edges = sorted(typed_scores.items(), key=lambda x: -x[1])
        if not sorted_edges:
            continue
        max_score = sorted_edges[0][1]
        if max_score <= 0:
            continue
        top: list[list] = []
        selected = _select_balanced_edges(sorted_edges, top_k) if balanced_lanes else sorted_edges[:top_k]
        for (rel_type, codon), score in selected:
            wq = int(round(score / max_score * 255))
            wq = max(1, min(255, wq))  # clamp to [1, 255] (0 is sentinel)
            top.append([rel_type, codon, wq])
        out[word] = top
    return out


_LANE_BY_RELATION = {
    "SYN": "lexical",
    "ISA": "lexical",
    "RELATED": "lexical",
    "ANTI": "lexical",
    "PART": "functional",
    "PROP": "functional",
    "USED_FOR": "functional",
    "CAUSES": "functional",
    "ENTAILS": "functional",
    "CTX": "functional",
    "ASSOC": "associative",
    "TOPIC": "associative",
}


def _select_balanced_edges(
    sorted_edges: list[tuple[tuple[str, str], float]],
    top_k: int,
) -> list[tuple[tuple[str, str], float]]:
    """Keep adapter lanes represented within the small relation-slot budget."""
    if top_k <= 0:
        return []

    budgets = {"lexical": 2, "functional": 1, "associative": 1}
    selected: list[tuple[tuple[str, str], float]] = []
    selected_keys: set[tuple[str, str]] = set()

    for lane, budget in budgets.items():
        if len(selected) >= top_k:
            break
        taken = 0
        for edge in sorted_edges:
            key = edge[0]
            if key in selected_keys:
                continue
            if _LANE_BY_RELATION.get(key[0], "lexical") != lane:
                continue
            selected.append(edge)
            selected_keys.add(key)
            taken += 1
            if taken >= budget or len(selected) >= top_k:
                break

    for edge in sorted_edges:
        if len(selected) >= top_k:
            break
        key = edge[0]
        if key in selected_keys:
            continue
        selected.append(edge)
        selected_keys.add(key)

    return sorted(selected, key=lambda x: -x[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--assertions", type=Path, default=DEFAULT_ASSERTIONS,
        help="Path to conceptnet-assertions-5.7.0.csv.gz",
    )
    parser.add_argument(
        "--numberbatch", type=Path, default=DEFAULT_NUMBERBATCH,
        help="Path to numberbatch-en-19.08.txt.gz; use --no-numberbatch to skip.",
    )
    parser.add_argument("--no-numberbatch", action="store_true")
    parser.add_argument("--numberbatch-top-k", type=int, default=4)
    parser.add_argument("--numberbatch-min-cosine", type=float, default=0.28)
    parser.add_argument("--numberbatch-scale", type=float, default=1.10)
    parser.add_argument("--numberbatch-fill-only", action="store_true")
    parser.add_argument(
        "--swow", type=Path, default=None,
        help="Path to extracted SWOW/NetSet directory with names.tsv and adjacency.tsv.",
    )
    parser.add_argument("--swow-relation", choices=("ASSOC", "TOPIC"), default="ASSOC")
    parser.add_argument("--swow-top-k", type=int, default=4)
    parser.add_argument("--swow-min-weight", type=float, default=2.0)
    parser.add_argument("--swow-scale", type=float, default=0.12)
    parser.add_argument("--swow-reverse-scale", type=float, default=0.65)
    parser.add_argument("--swow-fill-only", action="store_true")
    parser.add_argument("--balanced-lanes", action="store_true")
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

    if not args.no_numberbatch:
        if not args.numberbatch.exists():
            raise SystemExit(
                f"Numberbatch file not found: {args.numberbatch}\n"
                f"Download from "
                f"https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
                f"numberbatch-en-19.08.txt.gz"
            )
        print(f"Distilling Numberbatch 19.08 from {args.numberbatch} ...")
        nb_edges = augment_with_numberbatch(
            word_relations,
            args.numberbatch,
            w2c,
            top_k=args.numberbatch_top_k,
            min_cosine=args.numberbatch_min_cosine,
            scale=args.numberbatch_scale,
            fill_only=args.numberbatch_fill_only,
        )
        print(f"  added {nb_edges:,} Numberbatch-derived RELATED candidates")

    if args.swow is not None:
        print(f"Distilling SWOW associations from {args.swow} ...")
        augment_with_swow(
            word_relations,
            args.swow,
            w2c,
            relation_type=args.swow_relation,
            top_k=args.swow_top_k,
            min_weight=args.swow_min_weight,
            scale=args.swow_scale,
            reverse_scale=args.swow_reverse_scale,
            fill_only=args.swow_fill_only,
        )

    print(f"Selecting top-{args.top_k} relations per word …")
    top = select_top_relations(
        word_relations,
        top_k=args.top_k,
        balanced_lanes=args.balanced_lanes,
    )
    print(f"  {len(top):,} words have stamped relations")

    # Stamp typed `trel` field onto each codebook entry.
    for word, raw in codebook["entries"].items():
        rels = top.get(word.lower())
        if rels:
            raw["trel"] = rels
            raw.pop("rel", None)
        else:
            raw.pop("trel", None)
            raw.pop("rel", None)

    # Inflectional propagation: a variant like "running" or "happier" that
    # has no ConceptNet edges of its own inherits its base lemma's
    # relations. We use the codebook's morphology cache to find variant→
    # base edges. Any variant whose base has `trel` gets a copy.
    print("Propagating relations from base lemmas to inflectional variants …")
    morph_layer = _load_morph_cache(codebook)
    inflection_propagated = 0
    if morph_layer is not None:
        for variant, base, _pos in morph_layer.get("edges", []):
            v_raw = codebook["entries"].get(variant)
            if v_raw is None or "trel" in v_raw:
                continue
            base_raw = codebook["entries"].get(base)
            if base_raw is None or "trel" not in base_raw:
                continue
            # Deep-copy the rel list so independent edits don't alias.
            v_raw["trel"] = [list(r) for r in base_raw["trel"]]
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
