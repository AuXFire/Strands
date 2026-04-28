"""Build codon→codon adjacency from ConceptNet Numberbatch.

Supports two sources:
  * Numberbatch 19.08 English (loaded from a local .txt.gz, 517 k words).
    Higher Spearman ρ on standard benchmarks (the upstream README reports
    0.61–0.85). Default when --nb19 is set or NUMBERBATCH_19_PATH points
    to a readable file.
  * Numberbatch 17.06 multilingual (via gensim downloader). Fallback.

The adjacency build uses max-of-pairwise-cosines across each codon's
representative seed words to preserve specific-word relatedness signal
that centroid averaging would lose. Output is a sparse top-N graph
quantized to uint8 weights.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "strands" / "data" / "codebook_v0.1.0.json"
DEFAULT_OUTPUT = DEFAULT_INPUT
NB19_DEFAULT_PATH = Path("/tmp/numberbatch/nb19.08.txt.gz")


def _load_numberbatch_17():
    import gensim.downloader as api
    return api.load("conceptnet-numberbatch-17-06-300")


def _load_numberbatch_19(path: Path):
    """Load Numberbatch 19.08 .txt.gz into a {word: vector} dict.

    Format: first line ``"<vocab_size> <dim>"``, then one
    ``"word v1 v2 ... vN"`` line per entry. Bare words (not /c/en/-prefixed).
    """
    print(f"Loading Numberbatch 19.08 from {path} …")
    vecs: dict[str, np.ndarray] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        header = f.readline().split()
        vocab_size = int(header[0])
        dim = int(header[1])
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    print(f"  loaded {len(vecs):,} vectors (dim {dim})")
    return vecs, dim


class _NB17Adapter:
    """Wraps gensim KeyedVectors so .has() and .get() work like the
    19.08 dict."""

    def __init__(self, kv):
        self.kv = kv

    def get(self, word):
        key = f"/c/en/{word.replace(' ', '_')}"
        if key in self.kv:
            return self.kv[key]
        return None


class _NB19Adapter:
    def __init__(self, vecs):
        self.vecs = vecs

    def get(self, word):
        return self.vecs.get(word.replace(" ", "_"))


def _codon_str(d: str, c: int, n: int) -> str:
    return f"{d}{c:01X}{n:02X}"


def build_adjacency(
    codebook: dict,
    nb,
    *,
    seeds_per_codon: int = 8,
    neighbors_per_codon: int = 32,
    min_weight: float = 0.05,
) -> dict:
    print("Grouping entries by codon …")
    by_codon: dict[str, list[str]] = defaultdict(list)
    for word, raw in codebook["entries"].items():
        cs = _codon_str(raw["d"], raw["c"], raw["n"])
        by_codon[cs].append(word)
    print(f"  {len(by_codon)} distinct codons")

    print(f"Collecting up to {seeds_per_codon} representative-word vectors per codon …")
    codon_vecs: dict[str, np.ndarray] = {}
    for codon, words in by_codon.items():
        sorted_words = sorted(words, key=lambda w: (len(w), w))
        vecs = []
        for w in sorted_words:
            v = nb.get(w)
            if v is None:
                continue
            n = np.linalg.norm(v)
            if n > 0:
                vecs.append(v / n)
            if len(vecs) >= seeds_per_codon:
                break
        if vecs:
            codon_vecs[codon] = np.stack(vecs)
    print(f"  {len(codon_vecs)} codons with at least one representative vector")

    codon_list = sorted(codon_vecs.keys())
    n_codons = len(codon_list)
    print(f"Computing per-codon-pair MAX-of-word-cosines ({n_codons} codons) …")

    sim_rows: list[np.ndarray] = []
    for i, codon_a in enumerate(codon_list):
        va = codon_vecs[codon_a]
        row = np.zeros(n_codons, dtype=np.float32)
        for j, codon_b in enumerate(codon_list):
            if j == i:
                row[j] = -1.0
                continue
            vb = codon_vecs[codon_b]
            row[j] = float((va @ vb.T).max())
        sim_rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n_codons} codons processed")

    print("Selecting top-N neighbors per codon …")
    adjacency: dict[str, list[list]] = {}
    for i, codon in enumerate(codon_list):
        sims = sim_rows[i]
        top_idx = np.argpartition(-sims, min(neighbors_per_codon, n_codons - 1))[
            :neighbors_per_codon
        ]
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
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument(
        "--nb19", action="store_true",
        help="Use Numberbatch 19.08 (English) from --nb19-path. Default: 17.06.",
    )
    parser.add_argument("--nb19-path", type=Path, default=NB19_DEFAULT_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading codebook from {args.codebook}")
    with args.codebook.open("r", encoding="utf-8") as f:
        codebook = json.load(f)

    use_nb19 = args.nb19 or args.nb19_path.exists()
    if use_nb19:
        if not args.nb19_path.exists():
            raise SystemExit(
                f"Numberbatch 19.08 file not found: {args.nb19_path}\n"
                f"Download from https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
            )
        vecs, _ = _load_numberbatch_19(args.nb19_path)
        nb = _NB19Adapter(vecs)
        nb_version = "nb-19.08"
    else:
        nb = _NB17Adapter(_load_numberbatch_17())
        nb_version = "nb-17.06.300"

    from strands.build.cache import get_or_build, _hash
    seeds_hash = codebook.get("stats", {}).get("seeds_hash", "unknown")
    adj_input_hash = _hash({
        "seeds": seeds_hash,
        "nb": nb_version,
        "seeds_per_codon": args.seeds,
        "neighbors": args.neighbors,
        "min_weight": args.min_weight,
    })

    if args.no_cache:
        adjacency = build_adjacency(
            codebook, nb,
            seeds_per_codon=args.seeds,
            neighbors_per_codon=args.neighbors,
            min_weight=args.min_weight,
        )
    else:
        adjacency = get_or_build(
            "adjacency", adj_input_hash,
            lambda: build_adjacency(
                codebook, nb,
                seeds_per_codon=args.seeds,
                neighbors_per_codon=args.neighbors,
                min_weight=args.min_weight,
            ),
            cache_dir=args.cache_dir,
            sources={
                "seeds_hash": seeds_hash,
                "numberbatch": nb_version,
                "seeds_per_codon": args.seeds,
                "neighbors": args.neighbors,
                "min_weight": args.min_weight,
            },
        )

    codebook["codon_adjacency"] = adjacency
    codebook.setdefault("stats", {})["codon_adjacency_edges"] = sum(
        len(v) for v in adjacency.values()
    )
    codebook["stats"]["codon_adjacency_source"] = nb_version

    print(f"Writing codebook with adjacency to {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    size = args.output.stat().st_size
    print(f"  total codebook size: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
