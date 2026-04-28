#!/usr/bin/env python
"""Sentence STS section."""
import json
import re
import sys

import gensim.downloader as api
import numpy as np
from scipy.stats import spearmanr

from strands import compare
from tests.benchmarks._loaders import DATA_DIR, load_sick, load_sts


SENTENCE_DATASETS = {
    "STS-2012-MSRpar":    (DATA_DIR / "sts2012_msrpar.tsv", load_sts),
    "STS-2014-headlines": (DATA_DIR / "sts2014_headlines.tsv", load_sts),
    "STS-2014-images":    (DATA_DIR / "sts2014_images.tsv", load_sts),
    "STS-2015-headlines": (DATA_DIR / "sts2015_headlines.tsv", load_sts),
    "SICK-test":          (DATA_DIR / "sick_test.txt", load_sick),
}

_TOK = re.compile(r"[A-Za-z]+")


def cos(va, vb):
    if va is None or vb is None: return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def svec(m, s):
    toks = [t.lower() for t in _TOK.findall(s)]
    vs = [m[t] for t in toks if t in m]
    return np.mean(vs, axis=0) if vs else None


def main():
    g300 = api.load("glove-wiki-gigaword-300")
    rows = []
    for name, (path, loader) in SENTENCE_DATASETS.items():
        pairs = loader(path)
        gold = [g for _, _, g in pairs]
        sp = [compare(a, b, conceptnet_bridge=True).score for a, b, _ in pairs]
        gp = [cos(svec(g300, a), svec(g300, b)) for a, b, _ in pairs]
        rho_s, _ = spearmanr(gold, sp)
        rho_g, _ = spearmanr(gold, gp)
        rows.append({
            "name": name, "n": len(pairs),
            "strand": float(rho_s), "glove300": float(rho_g),
        })
    json.dump({"section": "B. Sentence STS", "rows": rows, "metric": "Spearman ρ"},
              sys.stdout)


if __name__ == "__main__":
    main()
