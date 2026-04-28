#!/usr/bin/env python
"""Cross-language algorithm fixture: within vs across separation."""
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

import gensim.downloader as api
import numpy as np

from strands import clone_similarity, encode_code

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "code_algorithms.json"
_TOK = re.compile(r"[A-Za-z]+")


def cos(va, vb):
    if va is None or vb is None: return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def svec(m, s):
    vs = [m[t.lower()] for t in _TOK.findall(s) if t.lower() in m]
    return np.mean(vs, axis=0) if vs else None


def main():
    g300 = api.load("glove-wiki-gigaword-300")
    data = json.loads(FIXTURE.read_text())["algorithms"]
    items = [(algo, lang, src) for algo, langs in data.items() for lang, src in langs.items()]

    strands = {(a, l): encode_code(s, language=l).strand for a, l, s in items}
    vecs = {(a, l): svec(g300, src) for a, l, src in items}

    s_within, s_across, g_within, g_across = [], [], [], []
    for (algo_a, lang_a, _), (algo_b, lang_b, _) in combinations(items, 2):
        s = clone_similarity(strands[(algo_a, lang_a)], strands[(algo_b, lang_b)])
        g = cos(vecs[(algo_a, lang_a)], vecs[(algo_b, lang_b)])
        if algo_a == algo_b and lang_a != lang_b:
            s_within.append(s); g_within.append(g)
        elif algo_a != algo_b:
            s_across.append(s); g_across.append(g)

    json.dump({
        "section": "E. Cross-language algorithm fixture",
        "rows": [{
            "name": "within - across",
            "n": len(s_within) + len(s_across),
            "strand": mean(s_within) - mean(s_across),
            "glove300": mean(g_within) - mean(g_across),
            "strand_within": mean(s_within), "strand_across": mean(s_across),
            "glove_within": mean(g_within), "glove_across": mean(g_across),
        }],
        "metric": "separation (higher = more discriminating)",
    }, sys.stdout)


if __name__ == "__main__":
    main()
