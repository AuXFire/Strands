#!/usr/bin/env python
"""Detailed diagnostic of strand weaknesses.

Runs strand vs GloVe on the standard SimLex-999 / WordSim-353 datasets,
then surfaces:
  1. Score-distribution histogram (the discrete-scoring ceiling).
  2. Per-quartile gold vs strand correlation (where does the disagreement live?)
  3. Worst false negatives (high human, zero strand) — likely missing seeds
     or split-domain concepts.
  4. Worst false positives (low human, high strand) — likely category
     overlaps that aren't real similarity.
  5. Out-of-vocab gaps and domain distribution skew.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import gensim.downloader as gensim_api
import numpy as np
from scipy.stats import spearmanr

from strands import compare, default_codebook
from strands.codebook import Codebook

ROOT = Path(__file__).parent.parent
DATA = ROOT / "tests" / "benchmarks" / "data"


def load_dataset(path: Path) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append((parts[0], parts[1], float(parts[2])))
    return out


def normalize_gold(values: list[float]) -> list[float]:
    """Scale gold scores to [0, 1] range for fair comparison."""
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) for v in values]


def strand_score(a: str, b: str) -> float:
    return compare(a, b).score


def glove_score(model, a: str, b: str) -> float | None:
    if a not in model or b not in model:
        return None
    va, vb = model[a], model[b]
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return None
    return float(np.dot(va, vb) / denom)


def header(text: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print("─" * 70)


def histogram(values: list[float], bins: int = 8, label: str = "") -> None:
    if not values:
        return
    arr = np.array(values)
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    width = max(counts)
    print(f"  {label} ({len(values)} values):")
    for i, c in enumerate(counts):
        bar = "█" * int(c / width * 40)
        print(f"    {edges[i]:5.2f}–{edges[i+1]:5.2f} | {c:4d} {bar}")


def analyze_dataset(name: str, dataset_path: Path, glove50, glove300):
    header(f"{name}")
    pairs = load_dataset(dataset_path)
    gold_raw = [g for _, _, g in pairs]
    gold = normalize_gold(gold_raw)

    strand_scores = [strand_score(a, b) for a, b, _ in pairs]
    g50_scores = [glove_score(glove50.model, a, b) or 0.0 for a, b, _ in pairs]
    g300_scores = [glove_score(glove300.model, a, b) or 0.0 for a, b, _ in pairs]

    rho_s, _ = spearmanr(gold, strand_scores)
    rho_50, _ = spearmanr(gold, g50_scores)
    rho_300, _ = spearmanr(gold, g300_scores)

    print(f"  Spearman ρ   strand={rho_s:.4f}  GloVe-50={rho_50:.4f}  "
          f"GloVe-300={rho_300:.4f}")

    # Discrete level histogram
    distinct = sorted(set(round(s, 4) for s in strand_scores))
    print(f"  Distinct strand scores: {len(distinct)}")
    counter = Counter(round(s, 2) for s in strand_scores)
    for level in sorted(counter):
        bar = "█" * int(counter[level] / max(counter.values()) * 30)
        print(f"    {level:.2f}  ({counter[level]:3d}) {bar}")

    # Per-quartile correlation
    print(f"\n  Per-gold-quartile correlation:")
    pairs_sorted = sorted(zip(pairs, strand_scores, gold), key=lambda x: x[2])
    n = len(pairs_sorted)
    for q in range(4):
        chunk = pairs_sorted[q * n // 4 : (q + 1) * n // 4]
        c_gold = [c[2] for c in chunk]
        c_strand = [c[1] for c in chunk]
        if len(set(c_strand)) < 2 or len(set(c_gold)) < 2:
            rho_q = float("nan")
        else:
            rho_q, _ = spearmanr(c_gold, c_strand)
        print(f"    Q{q+1} (gold {chunk[0][2]:.2f}–{chunk[-1][2]:.2f}): "
              f"ρ={rho_q:.3f}")

    # False negatives — humans see similarity, strands give 0 or near-0
    fn = sorted(
        [(a, b, g, s) for ((a, b, _), s, g) in pairs_sorted if g >= 0.6 and s <= 0.05],
        key=lambda x: -x[2],
    )[:8]
    if fn:
        print(f"\n  False NEGATIVES (humans similar, strand ≈ 0):")
        cb = default_codebook()
        for a, b, g, s in fn:
            ea = cb.lookup(a)
            eb = cb.lookup(b)
            ca = ea.codon.to_str() if ea else "?"
            cb_s = eb.codon.to_str() if eb else "?"
            print(f"    {a:14} {b:14} gold={g:.2f}  strand={s:.3f}  ({ca} vs {cb_s})")

    # False positives — humans see no similarity, strands give 0.4 (cat match)
    fp = sorted(
        [(a, b, g, s) for ((a, b, _), s, g) in pairs_sorted if g <= 0.3 and s >= 0.4],
        key=lambda x: -x[3],
    )[:8]
    if fp:
        print(f"\n  False POSITIVES (humans different, strand ≥ 0.4):")
        cb = default_codebook()
        for a, b, g, s in fp:
            ea = cb.lookup(a)
            eb = cb.lookup(b)
            ca = ea.codon.to_str() if ea else "?"
            cb_s = eb.codon.to_str() if eb else "?"
            print(f"    {a:14} {b:14} gold={g:.2f}  strand={s:.3f}  ({ca} vs {cb_s})")


def domain_distribution_check(cb: Codebook):
    """Are entries balanced across text domains?"""
    header("Domain distribution (text codebook)")
    counts: Counter = Counter()
    for word, raw in cb._entries.items():  # type: ignore[attr-defined]
        counts[raw["d"]] += 1
    total = sum(counts.values())
    for d, c in counts.most_common():
        share = c / total
        bar = "█" * int(share * 60)
        print(f"  {d}  {c:6,} ({share:5.1%})  {bar}")


def main() -> None:
    print("Loading models …")
    glove50 = type("M", (), {"model": gensim_api.load("glove-wiki-gigaword-50")})
    glove300 = type("M", (), {"model": gensim_api.load("glove-wiki-gigaword-300")})
    cb = default_codebook()
    print(f"Strand codebook: {len(cb):,} text + {cb.code_size} code entries")

    domain_distribution_check(cb)
    analyze_dataset("SimLex-999", DATA / "simlex999.txt", glove50, glove300)
    analyze_dataset("WordSim-353", DATA / "wordsim353.tsv", glove50, glove300)


if __name__ == "__main__":
    main()
