#!/usr/bin/env python
"""Sentence-level benchmark: strand vs GloVe-300 mean-of-vectors baseline.

Datasets: STS-2012 MSRpar, STS-2014 headlines+images, STS-2015 headlines, SICK.

For embeddings, sentence vector = mean of token vectors (a common baseline).
For strands, we use compare_strands which does greedy alignment over the
codon sequences.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gensim.downloader as api
import numpy as np
from scipy.stats import spearmanr

from strands import compare
from tests.benchmarks._loaders import DATA_DIR, load_sick, load_sts


_TOK_RE = re.compile(r"[A-Za-z]+")


DATASETS: dict[str, tuple[Path, callable]] = {
    "STS-2012-MSRpar":   (DATA_DIR / "sts2012_msrpar.tsv", load_sts),
    "STS-2014-headlines":(DATA_DIR / "sts2014_headlines.tsv", load_sts),
    "STS-2014-images":   (DATA_DIR / "sts2014_images.tsv", load_sts),
    "STS-2015-headlines":(DATA_DIR / "sts2015_headlines.tsv", load_sts),
    "SICK-test":         (DATA_DIR / "sick_test.txt", load_sick),
}


def sentence_vector(model, sentence: str) -> np.ndarray | None:
    tokens = [t.lower() for t in _TOK_RE.findall(sentence)]
    vecs = [model[t] for t in tokens if t in model]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def cosine(va: np.ndarray | None, vb: np.ndarray | None) -> float:
    if va is None or vb is None:
        return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    if d == 0:
        return 0.0
    return float(np.dot(va, vb) / d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conceptnet", action="store_true",
                        help="Enable strand ConceptNet bridge.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    args = parser.parse_args()

    print("Loading GloVe-300 …")
    g300 = api.load("glove-wiki-gigaword-300")
    print()

    print(f"{'Dataset':<22} {'pairs':>6} | {'Strand ρ':>10} {'GloVe-300 ρ':>13} | {'verdict':>16}")
    print("-" * 80)
    wins = losses = ties = 0

    for name in args.datasets:
        if name not in DATASETS:
            continue
        path, loader = DATASETS[name]
        pairs = loader(path)

        gold = [g for _, _, g in pairs]
        strand_pred = [
            compare(a, b).score  # pure strand-native; no flags
            for a, b, _ in pairs
        ]
        glove_pred = [
            cosine(sentence_vector(g300, a), sentence_vector(g300, b))
            for a, b, _ in pairs
        ]

        rho_s, _ = spearmanr(gold, strand_pred)
        rho_g, _ = spearmanr(gold, glove_pred)
        diff = rho_s - rho_g
        verdict = "WIN" if diff > 0.01 else "LOSE" if diff < -0.01 else "TIE"
        if verdict == "WIN":
            wins += 1
        elif verdict == "LOSE":
            losses += 1
        else:
            ties += 1
        print(f"{name:<22} {len(pairs):>6} | {rho_s:>10.4f} {rho_g:>13.4f} | {verdict:>10} ({diff:+.4f})")

    print("-" * 80)
    print(f"Strand vs GloVe-300: wins={wins}  losses={losses}  ties={ties}")


if __name__ == "__main__":
    main()
