#!/usr/bin/env python
"""CodeXGLUE WebQuery NL→Code retrieval (MRR)."""
import json
import re
import sys
from pathlib import Path
from statistics import mean

import gensim.downloader as api
import numpy as np

from strands import compare_strands, encode, encode_code

DATA = Path(__file__).parent.parent / "tests" / "benchmarks" / "data" / "codesearch_webquery.json"
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
    records = json.loads(DATA.read_text())[:100]

    code_strands = []
    for r in records:
        try:
            code_strands.append(encode_code(r["code"], language="python").strand)
        except Exception:
            code_strands.append(encode(r["code"]).strand)

    rrs_strand = []
    for i, r in enumerate(records):
        q = encode(r["doc"]).strand
        scores = [(compare_strands(q, c, conceptnet_bridge=True).score, j)
                  for j, c in enumerate(code_strands)]
        scores.sort(reverse=True)
        for rank, (_, j) in enumerate(scores, 1):
            if j == i:
                rrs_strand.append(1.0 / rank); break

    rrs_glove = []
    code_vecs = [svec(g300, r["code"]) for r in records]
    for i, r in enumerate(records):
        qv = svec(g300, r["doc"])
        scores = [(cos(qv, cv), j) for j, cv in enumerate(code_vecs)]
        scores.sort(reverse=True)
        for rank, (_, j) in enumerate(scores, 1):
            if j == i:
                rrs_glove.append(1.0 / rank); break

    json.dump({
        "section": "C. Code retrieval — CodeXGLUE WebQuery",
        "rows": [{"name": "CodeXGLUE WebQuery", "n": len(records),
                  "strand": mean(rrs_strand), "glove300": mean(rrs_glove)}],
        "metric": "MRR",
    }, sys.stdout)


if __name__ == "__main__":
    main()
