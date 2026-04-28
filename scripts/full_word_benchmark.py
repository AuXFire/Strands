#!/usr/bin/env python
"""Comprehensive word-similarity benchmark: strand vs GloVe-50/300 across all
available datasets. Reports ρ, coverage, and per-dataset winner."""

from __future__ import annotations

import argparse
from pathlib import Path

import gensim.downloader as gensim_api
import numpy as np
from scipy.stats import spearmanr

from strands import compare
from tests.benchmarks._loaders import (
    DATA_DIR,
    load_men3000,
    load_pairs_tsv,
    load_rg65,
    load_simverb,
)

DATASETS: dict[str, callable] = {
    "SimLex-999":   lambda: load_pairs_tsv(DATA_DIR / "simlex999.txt"),
    "WordSim-353":  lambda: load_pairs_tsv(DATA_DIR / "wordsim353.tsv"),
    "MEN-3000":     load_men3000,
    "RG-65":        load_rg65,
    "SimVerb-3500": load_simverb,
}


def strand_score(a: str, b: str, *, conceptnet: bool = False) -> float:
    # Pure strand-native compare; the conceptnet flag is ignored — kept
    # only for backwards-compatible CLI parity.
    return compare(a, b).score


def glove_cos(model, a: str, b: str) -> float | None:
    if a not in model or b not in model:
        return None
    va, vb = model[a], model[b]
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    if d == 0:
        return None
    return float(np.dot(va, vb) / d)


def run(model_name_to_loader, datasets, *, conceptnet: bool = False):
    rows: list[tuple] = []
    for ds_name, loader in datasets.items():
        pairs = loader()
        gold = [g for _, _, g in pairs]
        strand_pred = [strand_score(a, b, conceptnet=conceptnet) for a, b, _ in pairs]

        rho_strand, _ = spearmanr(gold, strand_pred)

        glove_results: dict[str, tuple[float, int]] = {}
        for mn, model in model_name_to_loader.items():
            preds = []
            cov = 0
            paired_gold = []
            for a, b, g in pairs:
                s = glove_cos(model, a, b)
                if s is None:
                    s = 0.0
                else:
                    cov += 1
                preds.append(s)
                paired_gold.append(g)
            rho, _ = spearmanr(paired_gold, preds)
            glove_results[mn] = (float(rho), cov)

        rows.append((ds_name, len(pairs), rho_strand, glove_results))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--conceptnet", action="store_true",
                        help="Enable ConceptNet/Numberbatch tier-3 bridge.")
    args = parser.parse_args()

    datasets = {k: DATASETS[k] for k in args.datasets if k in DATASETS}

    print("Loading GloVe …")
    g50 = gensim_api.load("glove-wiki-gigaword-50")
    g300 = gensim_api.load("glove-wiki-gigaword-300")
    print()
    rows = run({"GloVe-50": g50, "GloVe-300": g300}, datasets,
               conceptnet=args.conceptnet)

    print(f"{'Dataset':<14} {'pairs':>6} | {'Strand ρ':>10} {'GloVe-50 ρ':>12} {'GloVe-300 ρ':>13} | {'Strand vs G300':>16}")
    print("-" * 80)
    wins = losses = ties = 0
    for ds_name, n, rho_s, gres in rows:
        rho_50, cov50 = gres["GloVe-50"]
        rho_300, cov300 = gres["GloVe-300"]
        diff = rho_s - rho_300
        verdict = "WIN" if diff > 0.01 else "LOSE" if diff < -0.01 else "TIE"
        if verdict == "WIN":
            wins += 1
        elif verdict == "LOSE":
            losses += 1
        else:
            ties += 1
        print(f"{ds_name:<14} {n:>6} | {rho_s:>10.4f} {rho_50:>12.4f} {rho_300:>13.4f} | {verdict:>10} ({diff:+.4f})")
    print("-" * 80)
    print(f"Strand vs GloVe-300: wins={wins}  losses={losses}  ties={ties}")


if __name__ == "__main__":
    main()
