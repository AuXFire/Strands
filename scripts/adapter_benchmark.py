#!/usr/bin/env python
"""Compare adapter profiles on benchmark slices without changing official scores."""

from __future__ import annotations

from statistics import mean

from scipy.stats import spearmanr

from strands import compare
from tests.benchmarks._loaders import (
    DATA_DIR,
    load_men3000,
    load_pairs_tsv,
    load_rg65,
    load_simverb,
)


WORD_DATASETS = {
    "SimLex-999": lambda: load_pairs_tsv(DATA_DIR / "simlex999.txt"),
    "WordSim-353": lambda: load_pairs_tsv(DATA_DIR / "wordsim353.tsv"),
    "MEN-3000": load_men3000,
    "RG-65": load_rg65,
    "SimVerb-3500": load_simverb,
}


def word_section(profile: str) -> list[tuple[str, int, float]]:
    rows = []
    for name, loader in WORD_DATASETS.items():
        pairs = loader()
        gold = [g for _, _, g in pairs]
        scores = [compare(a, b, conceptnet_bridge=True, profile=profile).score for a, b, _ in pairs]
        rho, _ = spearmanr(gold, scores)
        rows.append((name, len(pairs), float(rho)))
    return rows


def main() -> None:
    for profile in ("auto", "strict", "topical"):
        rows = word_section(profile)
        print(f"\n{profile}")
        for name, n, score in rows:
            print(f"  {name:<14} {n:>5}  {score:.4f}")
        print(f"  {'average':<14}       {mean(score for _, _, score in rows):.4f}")


if __name__ == "__main__":
    main()
