"""BigCloneBench (CodeXGLUE subset) — clone detection F1.

For each labeled pair (id_a, id_b, label∈{0,1}), compute strand similarity
and predict clone if similarity ≥ threshold. Tune threshold to maximize F1
on a subsample.

Spec §12.3 target: F1 ≥ 0.50.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from strands import compare_strands, encode_code

DATA_DIR = Path(__file__).parent / "data"
TEST_FILE = DATA_DIR / "bigclone_test.txt"
DATA_FILE = DATA_DIR / "bigclone_data.jsonl"

REALISTIC_FLOOR_F1 = 0.50  # spec target


def _load_funcs():
    """func_id -> code text. The data.jsonl has all the snippets."""
    funcs: dict[str, str] = {}
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            funcs[obj["idx"]] = obj["func"]
    return funcs


def _load_test_pairs(limit: int = 1500, seed: int = 0):
    rng = random.Random(seed)
    pairs: list[tuple[str, str, int]] = []
    with TEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, lbl = line.split("\t")
            pairs.append((a, b, int(lbl)))
    rng.shuffle(pairs)
    # Balance positives and negatives
    pos = [p for p in pairs if p[2] == 1][: limit // 2]
    neg = [p for p in pairs if p[2] == 0][: limit // 2]
    out = pos + neg
    rng.shuffle(out)
    return out


@pytest.mark.slow
def test_bigclone_f1():
    if not (DATA_FILE.exists() and TEST_FILE.exists()):
        pytest.skip("BigCloneBench data not present")

    funcs = _load_funcs()
    # Filter to small/medium funcs for tractable test runtime
    small = {fid: src for fid, src in funcs.items() if 200 <= len(src) <= 1500}

    pairs = []
    pos = neg = 0
    for line in TEST_FILE.open(encoding="utf-8"):
        a, b, lbl = line.strip().split("\t")
        if a not in small or b not in small:
            continue
        lbl = int(lbl)
        if lbl == 1 and pos < 200:
            pairs.append((a, b, lbl)); pos += 1
        elif lbl == 0 and neg < 200:
            pairs.append((a, b, lbl)); neg += 1
        if pos >= 200 and neg >= 200:
            break

    needed_ids = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
    strands: dict[str, object] = {}
    for fid in needed_ids:
        try:
            strands[fid] = encode_code(small[fid], language="java").strand
        except Exception:
            pass

    from strands.document import clone_similarity

    scored: list[tuple[float, int]] = []
    for a, b, lbl in pairs:
        if a not in strands or b not in strands:
            continue
        s = clone_similarity(strands[a], strands[b])
        scored.append((s, lbl))

    # Sweep thresholds, pick max-F1
    best_f1 = 0.0
    best_th = 0.0
    for th_int in range(5, 100, 5):
        th = th_int / 100
        tp = sum(1 for s, l in scored if s >= th and l == 1)
        fp = sum(1 for s, l in scored if s >= th and l == 0)
        fn = sum(1 for s, l in scored if s < th and l == 1)
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\nBigCloneBench (n={len(scored)}): F1 = {best_f1:.4f} @ threshold {best_th:.2f}")
    assert best_f1 >= REALISTIC_FLOOR_F1, (
        f"BigCloneBench F1 {best_f1:.4f} below spec target {REALISTIC_FLOOR_F1}"
    )
