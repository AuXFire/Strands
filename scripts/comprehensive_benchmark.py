#!/usr/bin/env python
"""Comprehensive benchmark report: strand vs GloVe-50/300 across every
wired test in the project.

Sections:
  A. Word similarity      — SimLex-999, WordSim-353, MEN-3000, RG-65, SimVerb-3500
  B. Sentence STS         — STS-2012-MSRpar, STS-2014-headlines, STS-2014-images,
                             STS-2015-headlines, SICK-test
  C. Code retrieval (NL→Code) — CodeXGLUE WebQuery (MRR)
  D. Code clone detection      — BigCloneBench (F1)
  E. Cross-language algorithm fixture — internal (within vs across mean)
  F. Storage and encoding speed
"""

from __future__ import annotations

import json
import random
import re
import time
from itertools import combinations
from pathlib import Path
from statistics import mean

import gensim.downloader as api
import numpy as np
from scipy.stats import spearmanr

from strands import compare, compare_strands, encode, encode_code, default_codebook
from tests.benchmarks._loaders import (
    DATA_DIR,
    load_men3000,
    load_pairs_tsv,
    load_rg65,
    load_sick,
    load_simverb,
    load_sts,
)

WORD_DATASETS = {
    "SimLex-999":   lambda: load_pairs_tsv(DATA_DIR / "simlex999.txt"),
    "WordSim-353":  lambda: load_pairs_tsv(DATA_DIR / "wordsim353.tsv"),
    "MEN-3000":     load_men3000,
    "RG-65":        load_rg65,
    "SimVerb-3500": load_simverb,
}

SENTENCE_DATASETS = {
    "STS-2012-MSRpar":    (DATA_DIR / "sts2012_msrpar.tsv", load_sts),
    "STS-2014-headlines": (DATA_DIR / "sts2014_headlines.tsv", load_sts),
    "STS-2014-images":    (DATA_DIR / "sts2014_images.tsv", load_sts),
    "STS-2015-headlines": (DATA_DIR / "sts2015_headlines.tsv", load_sts),
    "SICK-test":          (DATA_DIR / "sick_test.txt", load_sick),
}

CODESEARCH_DATA = DATA_DIR / "codesearch_webquery.json"
BIGCLONE_DATA = DATA_DIR / "bigclone_data.jsonl"
BIGCLONE_TEST = DATA_DIR / "bigclone_test.txt"
ALGO_FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "code_algorithms.json"


def hr(title: str = "") -> None:
    print()
    if title:
        print(f"=== {title} " + "=" * (75 - len(title)))
    else:
        print("=" * 80)


def cos(va, vb) -> float:
    if va is None or vb is None:
        return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def glove_word_score(model, a: str, b: str) -> float:
    if a not in model or b not in model:
        return 0.0
    return cos(model[a], model[b])


def glove_sentence_vector(model, sentence: str):
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", sentence)]
    vecs = [model[t] for t in toks if t in model]
    return np.mean(vecs, axis=0) if vecs else None


def section_word(g50, g300):
    hr("A. Word similarity (Spearman ρ vs human gold)")
    print(f"{'Dataset':<14} {'pairs':>5} | {'Strand':>8} {'GloVe-50':>9} {'GloVe-300':>11} | {'verdict':>14}")
    print("-" * 80)
    rows = []
    for name, loader in WORD_DATASETS.items():
        pairs = loader()
        gold = [g for _, _, g in pairs]
        strand_p = [compare(a, b, conceptnet_bridge=True).score for a, b, _ in pairs]
        g50_p = [glove_word_score(g50, a, b) for a, b, _ in pairs]
        g300_p = [glove_word_score(g300, a, b) for a, b, _ in pairs]
        rho_s, _ = spearmanr(gold, strand_p)
        rho_50, _ = spearmanr(gold, g50_p)
        rho_300, _ = spearmanr(gold, g300_p)
        diff = rho_s - rho_300
        verdict = "WIN" if diff > 0.01 else "LOSE" if diff < -0.01 else "TIE"
        rows.append((name, rho_s, rho_50, rho_300, verdict))
        print(f"{name:<14} {len(pairs):>5} | {rho_s:>8.4f} {rho_50:>9.4f} {rho_300:>11.4f} | {verdict:>9} ({diff:+.3f})")
    return rows


def section_sentence(g300):
    hr("B. Sentence STS (Spearman ρ vs human gold)")
    print(f"{'Dataset':<22} {'pairs':>5} | {'Strand':>8} {'GloVe-300 mean':>16} | {'verdict':>14}")
    print("-" * 80)
    rows = []
    for name, (path, loader) in SENTENCE_DATASETS.items():
        pairs = loader(path)
        gold = [g for _, _, g in pairs]
        strand_p = [compare(a, b, conceptnet_bridge=True).score for a, b, _ in pairs]
        glove_p = [
            cos(glove_sentence_vector(g300, a), glove_sentence_vector(g300, b))
            for a, b, _ in pairs
        ]
        rho_s, _ = spearmanr(gold, strand_p)
        rho_g, _ = spearmanr(gold, glove_p)
        diff = rho_s - rho_g
        verdict = "WIN" if diff > 0.01 else "LOSE" if diff < -0.01 else "TIE"
        rows.append((name, rho_s, rho_g, verdict))
        print(f"{name:<22} {len(pairs):>5} | {rho_s:>8.4f} {rho_g:>16.4f} | {verdict:>9} ({diff:+.3f})")
    return rows


def section_code_search(g300):
    hr("C. Code retrieval — CodeXGLUE WebQuery NL→Code (MRR, higher = better)")
    if not CODESEARCH_DATA.exists():
        print("(skipped — dataset not present)")
        return None
    records = json.loads(CODESEARCH_DATA.read_text())[:100]

    code_strands = []
    for r in records:
        try:
            code_strands.append(encode_code(r["code"], language="python").strand)
        except Exception:
            code_strands.append(encode(r["code"]).strand)

    # Strand MRR
    rrs_strand = []
    for i, r in enumerate(records):
        q = encode(r["doc"]).strand
        scores = [(compare_strands(q, c, conceptnet_bridge=True).score, j)
                  for j, c in enumerate(code_strands)]
        scores.sort(reverse=True)
        for rank, (_, j) in enumerate(scores, 1):
            if j == i:
                rrs_strand.append(1.0 / rank)
                break

    # GloVe MRR
    rrs_glove = []
    code_vecs = [glove_sentence_vector(g300, r["code"]) for r in records]
    for i, r in enumerate(records):
        qv = glove_sentence_vector(g300, r["doc"])
        scores = [(cos(qv, cv), j) for j, cv in enumerate(code_vecs)]
        scores.sort(reverse=True)
        for rank, (_, j) in enumerate(scores, 1):
            if j == i:
                rrs_glove.append(1.0 / rank)
                break

    mrr_s = mean(rrs_strand)
    mrr_g = mean(rrs_glove)
    verdict = "WIN" if mrr_s > mrr_g + 0.01 else "LOSE" if mrr_s < mrr_g - 0.01 else "TIE"
    print(f"{'CodeXGLUE WebQuery':<22} {len(records):>5} | {mrr_s:>8.4f} {mrr_g:>16.4f} | {verdict:>9} ({mrr_s - mrr_g:+.3f})")
    print("  (spec §12.3 target: MRR ≥ 0.25)")
    return [("CodeXGLUE WebQuery", mrr_s, mrr_g, verdict)]


def section_clone(g300):
    hr("D. Code clone detection — BigCloneBench subset (F1, higher = better)")
    if not (BIGCLONE_DATA.exists() and BIGCLONE_TEST.exists()):
        print("(skipped — dataset not present)")
        return None

    funcs: dict[str, str] = {}
    with BIGCLONE_DATA.open() as f:
        for line in f:
            obj = json.loads(line)
            if 200 <= len(obj["func"]) <= 1500:
                funcs[obj["idx"]] = obj["func"]

    pairs = []
    pos = neg = 0
    target = 50  # 50 pos + 50 neg = 100 total for tractable runtime
    for line in BIGCLONE_TEST.open():
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        a, b, lbl = parts[0], parts[1], int(parts[2])
        if a not in funcs or b not in funcs:
            continue
        if lbl == 1 and pos < target:
            pairs.append((a, b, lbl)); pos += 1
        elif lbl == 0 and neg < target:
            pairs.append((a, b, lbl)); neg += 1
        if pos >= target and neg >= target:
            break

    # Strand
    strands = {fid: encode_code(funcs[fid], language="java").strand
               for fid in {a for a, _, _ in pairs} | {b for _, b, _ in pairs}}
    s_scored = [(compare_strands(strands[a], strands[b], code_aware=True).score, lbl)
                for a, b, lbl in pairs]
    # GloVe
    vecs = {fid: glove_sentence_vector(g300, funcs[fid])
            for fid in strands.keys()}
    g_scored = [(cos(vecs[a], vecs[b]), lbl) for a, b, lbl in pairs]

    def best_f1(scored):
        best = 0.0
        best_th = 0.0
        for th_int in range(5, 100, 5):
            th = th_int / 100
            tp = sum(1 for s, l in scored if s >= th and l == 1)
            fp = sum(1 for s, l in scored if s >= th and l == 0)
            fn = sum(1 for s, l in scored if s < th and l == 1)
            if tp == 0:
                continue
            p = tp / (tp + fp); r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best:
                best = f1; best_th = th
        return best, best_th

    f1_s, th_s = best_f1(s_scored)
    f1_g, th_g = best_f1(g_scored)
    verdict = "WIN" if f1_s > f1_g + 0.01 else "LOSE" if f1_s < f1_g - 0.01 else "TIE"
    print(f"{'BigCloneBench':<22} {len(pairs):>5} | {f1_s:>8.4f} {f1_g:>16.4f} | {verdict:>9} ({f1_s - f1_g:+.3f})")
    print(f"  (best thresholds: strand={th_s:.2f}, glove={th_g:.2f}; spec §12.3 target F1 ≥ 0.50)")
    return [("BigCloneBench", f1_s, f1_g, verdict)]


def section_crosslang(g300):
    hr("E. Cross-language algorithm fixture (within > across)")
    data = json.loads(ALGO_FIXTURE.read_text())["algorithms"]
    items = []
    for algo, langs in data.items():
        for lang, src in langs.items():
            items.append((algo, lang, src))

    # Strand
    strands = {(a, l): encode_code(s, language=l).strand for a, l, s in items}
    s_within = []; s_across = []
    for (algo_a, lang_a, _), (algo_b, lang_b, _) in combinations(items, 2):
        sc = compare_strands(strands[(algo_a, lang_a)], strands[(algo_b, lang_b)],
                             code_aware=True).score
        if algo_a == algo_b and lang_a != lang_b:
            s_within.append(sc)
        elif algo_a != algo_b:
            s_across.append(sc)
    s_w, s_a = mean(s_within), mean(s_across)
    s_sep = s_w - s_a

    # GloVe
    vecs = {(a, l): glove_sentence_vector(g300, src) for a, l, src in items}
    g_within = []; g_across = []
    for (algo_a, lang_a, _), (algo_b, lang_b, _) in combinations(items, 2):
        sc = cos(vecs[(algo_a, lang_a)], vecs[(algo_b, lang_b)])
        if algo_a == algo_b and lang_a != lang_b:
            g_within.append(sc)
        elif algo_a != algo_b:
            g_across.append(sc)
    g_w, g_a = mean(g_within), mean(g_across)
    g_sep = g_w - g_a

    verdict = "WIN" if s_sep > g_sep + 0.01 else "LOSE" if s_sep < g_sep - 0.01 else "TIE"
    print(f"{'within mean':<14} | strand={s_w:.4f}  glove={g_w:.4f}")
    print(f"{'across mean':<14} | strand={s_a:.4f}  glove={g_a:.4f}")
    print(f"{'separation':<14} | strand={s_sep:.4f}  glove={g_sep:.4f}    {verdict} ({s_sep - g_sep:+.3f})")
    return [("Cross-lang separation", s_sep, g_sep, verdict)]


def section_storage_speed():
    hr("F. Storage and encoding speed")
    cb = default_codebook()
    print(f"  Strand codebook   : {len(cb):>10,} text + {cb.code_size:>4} code entries")
    print(f"  Strand storage    : 4 bytes/word (default), 8 bytes/word (v2 extended)")
    print(f"  GloVe-50          : 200 bytes/word (50 dim × 4-byte float)")
    print(f"  GloVe-300         : 1,200 bytes/word")
    print(f"  OpenAI 3-small    : 6,144 bytes/word (1536 dim × 4-byte float)")

    sample = ["happy", "joyful", "computer", "fetch", "running", "elephant",
              "philosophy", "asynchronous", "hereby", "pneumonia"] * 50
    t0 = time.perf_counter()
    for w in sample:
        encode(w)
    dt = time.perf_counter() - t0
    print(f"\n  Strand encode rate: {len(sample)/dt:>10,.0f} tokens/sec ({dt*1e6/len(sample):.1f} µs/token)")


def main():
    print("Loading GloVe …")
    g50 = api.load("glove-wiki-gigaword-50")
    g300 = api.load("glove-wiki-gigaword-300")

    word_rows = section_word(g50, g300)
    sentence_rows = section_sentence(g300)
    code_rows = section_code_search(g300) or []
    clone_rows = section_clone(g300) or []
    cross_rows = section_crosslang(g300) or []
    section_storage_speed()

    hr("SUMMARY — Strand vs GloVe-300")
    all_rows = []
    for name, rho_s, _, rho_300, verdict in word_rows:
        all_rows.append((f"  Word: {name}", rho_s, rho_300, verdict))
    for name, rho_s, rho_300, verdict in sentence_rows:
        all_rows.append((f"  Sent: {name}", rho_s, rho_300, verdict))
    for name, mrr_s, mrr_g, verdict in code_rows:
        all_rows.append((f"  Code: {name}", mrr_s, mrr_g, verdict))
    for name, f1_s, f1_g, verdict in clone_rows:
        all_rows.append((f"  Code: {name}", f1_s, f1_g, verdict))
    for name, sep_s, sep_g, verdict in cross_rows:
        all_rows.append((f"  Code: {name}", sep_s, sep_g, verdict))

    wins = sum(1 for r in all_rows if r[3] == "WIN")
    losses = sum(1 for r in all_rows if r[3] == "LOSE")
    ties = sum(1 for r in all_rows if r[3] == "TIE")
    print(f"\n  TOTAL: {len(all_rows)} benchmarks  |  WINS={wins}  TIES={ties}  LOSSES={losses}")

    avg_s = mean(r[1] for r in all_rows)
    avg_g = mean(r[2] for r in all_rows)
    print(f"  Average score: strand={avg_s:.4f}   competitor={avg_g:.4f}   diff={avg_s-avg_g:+.4f}")


if __name__ == "__main__":
    main()
