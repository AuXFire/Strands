#!/usr/bin/env python
"""Word similarity section — outputs one JSON record."""
import json
import sys

import gensim.downloader as api
import numpy as np
from scipy.stats import spearmanr

from strands import compare
from tests.benchmarks._loaders import (
    DATA_DIR, load_men3000, load_pairs_tsv, load_rg65, load_simverb,
)

WORD_DATASETS = {
    "SimLex-999":   lambda: load_pairs_tsv(DATA_DIR / "simlex999.txt"),
    "WordSim-353":  lambda: load_pairs_tsv(DATA_DIR / "wordsim353.tsv"),
    "MEN-3000":     load_men3000,
    "RG-65":        load_rg65,
    "SimVerb-3500": load_simverb,
}


def cos(va, vb):
    if va is None or vb is None: return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def gscore(model, a, b):
    if a not in model or b not in model: return 0.0
    return cos(model[a], model[b])


def main():
    g50 = api.load("glove-wiki-gigaword-50")
    g300 = api.load("glove-wiki-gigaword-300")
    rows = []
    for name, loader in WORD_DATASETS.items():
        pairs = loader()
        gold = [g for _, _, g in pairs]
        sp = [compare(a, b, conceptnet_bridge=True).score for a, b, _ in pairs]
        gp50 = [gscore(g50, a, b) for a, b, _ in pairs]
        gp300 = [gscore(g300, a, b) for a, b, _ in pairs]
        rho_s, _ = spearmanr(gold, sp)
        rho_50, _ = spearmanr(gold, gp50)
        rho_300, _ = spearmanr(gold, gp300)
        rows.append({
            "name": name, "n": len(pairs),
            "strand": float(rho_s), "glove50": float(rho_50), "glove300": float(rho_300),
        })
    json.dump({"section": "A. Word similarity", "rows": rows, "metric": "Spearman ρ"},
              sys.stdout)


if __name__ == "__main__":
    main()
