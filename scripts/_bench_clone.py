#!/usr/bin/env python
"""BigCloneBench clone detection F1."""
import json
import re
import sys
from pathlib import Path

import gensim.downloader as api
import numpy as np

from strands import compare_strands, encode_code

DATA_DIR = Path(__file__).parent.parent / "tests" / "benchmarks" / "data"
DATA_FILE = DATA_DIR / "bigclone_data.jsonl"
TEST_FILE = DATA_DIR / "bigclone_test.txt"
_TOK = re.compile(r"[A-Za-z]+")


def cos(va, vb):
    if va is None or vb is None: return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def svec(m, s):
    vs = [m[t.lower()] for t in _TOK.findall(s) if t.lower() in m]
    return np.mean(vs, axis=0) if vs else None


def best_f1(scored):
    best = 0.0; best_th = 0.0
    for th_int in range(5, 100, 5):
        th = th_int / 100
        tp = sum(1 for s, l in scored if s >= th and l == 1)
        fp = sum(1 for s, l in scored if s >= th and l == 0)
        fn = sum(1 for s, l in scored if s < th and l == 1)
        if tp == 0: continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best:
            best = f1; best_th = th
    return best, best_th


def main():
    g300 = api.load("glove-wiki-gigaword-300")

    funcs = {}
    with DATA_FILE.open() as f:
        for line in f:
            obj = json.loads(line)
            if 200 <= len(obj["func"]) <= 1500:
                funcs[obj["idx"]] = obj["func"]

    pairs = []
    pos = neg = 0
    target = 100
    for line in TEST_FILE.open():
        a, b, lbl = line.strip().split("\t")
        if a not in funcs or b not in funcs: continue
        lbl = int(lbl)
        if lbl == 1 and pos < target: pairs.append((a, b, lbl)); pos += 1
        elif lbl == 0 and neg < target: pairs.append((a, b, lbl)); neg += 1
        if pos >= target and neg >= target: break

    needed = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
    strands = {fid: encode_code(funcs[fid], language="java").strand for fid in needed}
    s_scored = [(compare_strands(strands[a], strands[b], code_aware=True).score, lbl)
                for a, b, lbl in pairs]

    vecs = {fid: svec(g300, funcs[fid]) for fid in needed}
    g_scored = [(cos(vecs[a], vecs[b]), lbl) for a, b, lbl in pairs]

    f1_s, th_s = best_f1(s_scored)
    f1_g, th_g = best_f1(g_scored)

    json.dump({
        "section": "D. Code clone detection — BigCloneBench",
        "rows": [{"name": "BigCloneBench", "n": len(pairs),
                  "strand": f1_s, "glove300": f1_g,
                  "strand_threshold": th_s, "glove_threshold": th_g}],
        "metric": "F1",
    }, sys.stdout)


if __name__ == "__main__":
    main()
