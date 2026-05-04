"""Build the backbone binary tables from source databases.

For Milestone 1, ingests WordNet synsets (nodes + WordNet relations) plus
ConceptNet 5.7 raw assertions (edges between codebook-mapped concepts).
Outputs four files:

  backbone.nodes      raw uint8 bytes of the (N,) np.ndarray of NODE_DTYPE
  backbone.edges      raw uint8 bytes of the (E,) np.ndarray of EDGE_DTYPE
  backbone.lemmas     utf-8 null-separated lemma strings; offsets in nodes
  backbone.manifest.json  metadata: counts, version, source attributions

Determinism: all iteration orders are sorted (synset names, lemma lists,
ConceptNet edges by source URI). Same inputs → byte-identical outputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from strands.backbone.schema import (
    EDGE_DTYPE,
    NODE_DTYPE,
    ConceptType,
    Rel,
    Source,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "strands" / "data" / "backbone"
DEFAULT_CODEBOOK = REPO_ROOT / "strands" / "data" / "codebook_v0.1.0.json"
DEFAULT_CONCEPTNET = Path("/tmp/conceptnet/assertions.csv.gz")


# ---- WordNet ingestion ---------------------------------------------------


_POS_TO_CONCEPT_TYPE = {
    "n": ConceptType.ENTITY,
    "v": ConceptType.EVENT,
    "a": ConceptType.PROPERTY,
    "s": ConceptType.PROPERTY,
    "r": ConceptType.PROPERTY,
}


def _pos_to_concept_type(pos: str) -> int:
    return _POS_TO_CONCEPT_TYPE.get(pos, ConceptType.ABSTRACT)


def _activation_default_for_synset(synset) -> int:
    """Heuristic default activation: more frequent → higher activation.
    Uses NLTK's lemma frequency counts. Returns uint16."""
    try:
        total = sum(l.count() for l in synset.lemmas())
    except Exception:
        total = 0
    return min(0xFFFF, max(1, int(total) * 64))


def _embedding_lsh(synset_name: str) -> bytes:
    """16-byte LSH-style digest of the synset name. Acts as a stable
    locality hash; not a proper LSH but deterministic and useful for
    quick first-pass equality checks."""
    return hashlib.blake2s(synset_name.encode("utf-8"), digest_size=16).digest()


# ---- ConceptNet ingestion ------------------------------------------------


_EN_PREFIX = "/c/en/"
_CONCEPT_RE = re.compile(r"^/c/en/([^/]+)")


_CONCEPTNET_TO_REL = {
    "/r/Synonym":             Rel.SYNONYM,
    "/r/Antonym":             Rel.ANTONYM,
    "/r/SimilarTo":           Rel.SIMILAR_TO,
    "/r/RelatedTo":           Rel.RELATED_TO,
    "/r/IsA":                 Rel.HYPERNYM,
    "/r/InstanceOf":          Rel.INSTANCE_OF,
    "/r/HasA":                Rel.HAS_A,
    "/r/PartOf":              Rel.PART_OF,
    "/r/MadeOf":              Rel.MADE_OF,
    "/r/UsedFor":             Rel.USED_FOR,
    "/r/CapableOf":           Rel.CAPABLE_OF,
    "/r/HasContext":          Rel.HAS_CONTEXT,
    "/r/AtLocation":          Rel.AT_LOCATION,
    "/r/HasProperty":         Rel.HAS_PROPERTY,
    "/r/HasPrerequisite":     Rel.HAS_PREREQUISITE,
    "/r/HasSubevent":         Rel.HAS_SUBEVENT,
    "/r/HasFirstSubevent":    Rel.HAS_FIRST_SUBEVENT,
    "/r/HasLastSubevent":     Rel.HAS_LAST_SUBEVENT,
    "/r/Causes":              Rel.CAUSES,
    "/r/CausesDesire":        Rel.DESIRES,
    "/r/MotivatedByGoal":     Rel.MOTIVATED_BY_GOAL,
    "/r/DerivedFrom":         Rel.DERIVED_FROM,
    "/r/MannerOf":            Rel.MANNER_OF,
    "/r/DefinedAs":           Rel.DEFINED_AS,
    "/r/FormOf":              Rel.FORM_OF,
}


def _extract_lemma(uri: str) -> str | None:
    m = _CONCEPT_RE.match(uri)
    if not m:
        return None
    return m.group(1).replace("_", " ")


# ---- Build pipeline ------------------------------------------------------


def _load_codebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _word_to_codon(codebook: dict) -> dict[str, tuple[str, int, int]]:
    """word → (domain_code_str, category, concept) for codon lookup."""
    out: dict[str, tuple[str, int, int]] = {}
    for word, raw in codebook.get("entries", {}).items():
        out[word.lower()] = (raw["d"], raw["c"], raw["n"])
    return out


def _domain_str_to_id(d: str) -> int:
    from strands.codon import DOMAIN_CODES
    return DOMAIN_CODES.get(d, 0xFF)


def build(
    out_dir: Path,
    *,
    codebook_path: Path = DEFAULT_CODEBOOK,
    conceptnet_path: Path | None = DEFAULT_CONCEPTNET,
    max_synsets: int | None = None,
) -> dict:
    """Build the backbone binary tables and write to ``out_dir``."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as e:
        raise SystemExit("nltk is required") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading codebook for codon assignments …")
    codebook = _load_codebook(codebook_path)
    word2codon = _word_to_codon(codebook)
    print(f"  {len(word2codon):,} words mapped to codons")

    # ---- Pass 1: enumerate synsets, assign node IDs ---------------------
    print("Enumerating WordNet synsets …")
    all_synsets = sorted(wn.all_synsets(), key=lambda s: s.name())
    if max_synsets is not None:
        all_synsets = all_synsets[:max_synsets]
    n_synsets = len(all_synsets)
    print(f"  {n_synsets:,} synsets")

    synset_to_id: dict[str, int] = {
        s.name(): i for i, s in enumerate(all_synsets)
    }

    # ---- Pass 2: build lemma table + lemma_offset per node ---------------
    print("Building lemma table …")
    lemma_buf = bytearray()
    node_lemma_offsets: list[int] = [0] * n_synsets
    node_lemma_counts: list[int] = [0] * n_synsets
    for i, synset in enumerate(all_synsets):
        lemmas = sorted({l.name().replace("_", " ").lower() for l in synset.lemmas()})
        node_lemma_offsets[i] = len(lemma_buf)
        node_lemma_counts[i] = min(255, len(lemmas))
        for lemma in lemmas[:255]:
            lemma_buf.extend(lemma.encode("utf-8"))
            lemma_buf.append(0)  # null separator
    print(f"  lemma table: {len(lemma_buf) / 1024:.1f} KB")

    # ---- Pass 3: build edges from WordNet relations ----------------------
    print("Building WordNet edges …")
    edges_buffer: list[tuple] = []  # (source_id, target_id, rel, weight, src_attr)

    def _add_wn_edge(src_id, target_synset, rel, weight=0xC000):
        if target_synset.name() not in synset_to_id:
            return
        edges_buffer.append((
            src_id, synset_to_id[target_synset.name()], int(rel),
            weight, Source.WORDNET,
        ))

    for i, synset in enumerate(all_synsets):
        for h in synset.hypernyms():
            _add_wn_edge(i, h, Rel.HYPERNYM)
        for h in synset.instance_hypernyms():
            _add_wn_edge(i, h, Rel.INSTANCE_OF)
        for h in synset.hyponyms():
            _add_wn_edge(i, h, Rel.HYPONYM)
        for m in synset.part_meronyms() + synset.member_meronyms():
            _add_wn_edge(i, m, Rel.MERONYM)
        for h in synset.part_holonyms() + synset.member_holonyms():
            _add_wn_edge(i, h, Rel.HOLONYM)
        for sim in synset.similar_tos():
            _add_wn_edge(i, sim, Rel.SIMILAR_TO)
        # Antonyms come via lemma pairs.
        for l in synset.lemmas():
            for a in l.antonyms():
                _add_wn_edge(i, a.synset(), Rel.ANTONYM, weight=0xA000)
    print(f"  WordNet edges: {len(edges_buffer):,}")

    # ---- Pass 4: ConceptNet edges (lemma-string-keyed) -------------------
    cn_edges = 0
    if conceptnet_path and conceptnet_path.exists():
        print(f"Streaming ConceptNet from {conceptnet_path} …")
        # Build first-lemma-per-synset reverse map.
        lemma_to_synset_id: dict[str, int] = {}
        for synset_name, sid in synset_to_id.items():
            try:
                first_lemma = wn.synset(synset_name).lemmas()[0].name().replace("_", " ").lower()
            except Exception:
                continue
            if first_lemma not in lemma_to_synset_id:
                lemma_to_synset_id[first_lemma] = sid

        with gzip.open(conceptnet_path, "rt", encoding="utf-8") as f:
            scanned = 0
            for line in f:
                scanned += 1
                try:
                    _, rel_uri, src_uri, tgt_uri, meta_json = line.rstrip("\n").split("\t", 4)
                except ValueError:
                    continue
                if not (src_uri.startswith(_EN_PREFIX) and tgt_uri.startswith(_EN_PREFIX)):
                    continue
                rel_id = _CONCEPTNET_TO_REL.get(rel_uri)
                if rel_id is None:
                    continue
                lemma_a = _extract_lemma(src_uri)
                lemma_b = _extract_lemma(tgt_uri)
                if not lemma_a or not lemma_b or lemma_a == lemma_b:
                    continue
                a_id = lemma_to_synset_id.get(lemma_a)
                b_id = lemma_to_synset_id.get(lemma_b)
                if a_id is None or b_id is None:
                    continue
                try:
                    weight = float(json.loads(meta_json).get("weight", 1.0))
                except (json.JSONDecodeError, ValueError):
                    weight = 1.0
                w16 = min(0xFFFF, max(0, int(weight * 16384)))
                edges_buffer.append((a_id, b_id, int(rel_id), w16, Source.CONCEPTNET))
                cn_edges += 1
                if cn_edges % 200_000 == 0:
                    print(f"  ... {cn_edges:,} ConceptNet edges added (scanned {scanned:,})")
        print(f"  ConceptNet edges: {cn_edges:,}")

    # ---- Pass 5: sort edges by source_id, build edge offsets ------------
    print("Sorting + indexing edges …")
    edges_buffer.sort(key=lambda e: (e[0], e[2]))  # by source_id, then relation
    n_edges = len(edges_buffer)

    edge_array = np.zeros(n_edges, dtype=EDGE_DTYPE)
    node_edge_offsets = [0] * n_synsets
    node_edge_counts = [0] * n_synsets
    cur_src = -1
    for idx, (src_id, tgt_id, rel, weight, src_attr) in enumerate(edges_buffer):
        if src_id != cur_src:
            node_edge_offsets[src_id] = idx
            cur_src = src_id
        node_edge_counts[src_id] += 1
        e = edge_array[idx]
        e["target_id"] = tgt_id
        e["relation_type"] = rel
        e["weight"] = weight
        e["confidence"] = 0xC000 if src_attr == Source.WORDNET else 0x8000
        e["context_volatility"] = 0
        e["source_attribution"] = src_attr
        e["bidirectional_flag"] = 1 if rel in (
            Rel.SIMILAR_TO, Rel.SYNONYM, Rel.RELATED_TO, Rel.ANTONYM
        ) else 0
        e["compute_on_conflict"] = 0

    # ---- Pass 6: build node array ---------------------------------------
    print("Building node array …")
    node_array = np.zeros(n_synsets, dtype=NODE_DTYPE)
    for i, synset in enumerate(all_synsets):
        n = node_array[i]
        n["node_id"] = i
        n["concept_type"] = _pos_to_concept_type(synset.pos())
        n["activation_default"] = _activation_default_for_synset(synset)
        n["volatility_flag"] = 0
        n["embedding_compressed"] = _embedding_lsh(synset.name())
        n["lemma_count"] = node_lemma_counts[i]
        n["lemma_offset"] = node_lemma_offsets[i]
        n["relationship_count"] = node_edge_counts[i]
        n["relationship_offset"] = node_edge_offsets[i]
        n["frame_id"] = 0
        n["language_independent_id"] = i
        # Codon link: try the first lemma against the codebook.
        try:
            first_lemma = synset.lemmas()[0].name().replace("_", " ").lower()
        except Exception:
            first_lemma = ""
        codon = word2codon.get(first_lemma)
        if codon is not None:
            d_str, c, conc = codon
            n["codon_domain"] = _domain_str_to_id(d_str)
            n["codon_category"] = c
            n["codon_concept"] = conc
        else:
            n["codon_domain"] = 0xFF  # unset

    # ---- Pass 7: serialize ----------------------------------------------
    print("Writing binary tables …")
    nodes_path = out_dir / "backbone.nodes"
    edges_path = out_dir / "backbone.edges"
    lemmas_path = out_dir / "backbone.lemmas"
    manifest_path = out_dir / "backbone.manifest.json"

    nodes_path.write_bytes(node_array.tobytes(order="C"))
    edges_path.write_bytes(edge_array.tobytes(order="C"))
    lemmas_path.write_bytes(bytes(lemma_buf))

    manifest = {
        "version": "0.1.0",
        "schema_version": 1,
        "node_count": int(n_synsets),
        "edge_count": int(n_edges),
        "lemma_buffer_bytes": int(len(lemma_buf)),
        "node_dtype_size": NODE_DTYPE.itemsize,
        "edge_dtype_size": EDGE_DTYPE.itemsize,
        "sources": ["wordnet-3.0", "conceptnet-5.7.0"] if cn_edges else ["wordnet-3.0"],
        "synset_id_map_hash": hashlib.sha256(
            "\n".join(s.name() for s in all_synsets).encode("utf-8")
        ).hexdigest()[:16],
        "files": {
            "nodes": nodes_path.name,
            "edges": edges_path.name,
            "lemmas": lemmas_path.name,
        },
        "stats": {
            "wordnet_edges": n_edges - cn_edges,
            "conceptnet_edges": cn_edges,
            "nodes_with_codon": int(np.sum(node_array["codon_domain"] != 0xFF)),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"  nodes: {nodes_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  edges: {edges_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  lemmas: {lemmas_path.stat().st_size / 1024 / 1024:.1f} MB")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--conceptnet", type=Path, default=DEFAULT_CONCEPTNET)
    parser.add_argument(
        "--max-synsets", type=int, default=None,
        help="Cap synset ingest for fast tests; None = all WordNet.",
    )
    parser.add_argument(
        "--no-conceptnet", action="store_true",
        help="Skip ConceptNet ingestion (WordNet only).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    manifest = build(
        args.out_dir,
        codebook_path=args.codebook,
        conceptnet_path=None if args.no_conceptnet else args.conceptnet,
        max_synsets=args.max_synsets,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {manifest['node_count']:,} nodes, "
          f"{manifest['edge_count']:,} edges "
          f"({manifest['stats']['nodes_with_codon']:,} with codon link)")


if __name__ == "__main__":
    main()
