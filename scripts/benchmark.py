#!/usr/bin/env python
"""Quick benchmark: Semantic Strands vs GloVe embeddings.

Measures three axes:
  1. Storage:   bytes per word.
  2. Encoding:  encode time per word (warm cache).
  3. Quality:   Spearman ρ on word-similarity datasets.

Both backends are evaluated on the same word lists. Embeddings use cosine
similarity; strands use the spec §8.1 alignment scorer (degenerate to a
single-codon comparison for word-level pairs).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import gensim.downloader as gensim_api

from strands.codebook import default_codebook
from strands.encoder import encode as strand_encode

# --- Pair sources ----------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


def load_fixture_pairs() -> list[tuple[str, str, float]]:
    """Combine our own synonym/antonym/unrelated fixtures into a graded scale.

    Synonym=1.0, antonym=0.5, unrelated=0.0 — a human-judgment proxy.
    """
    out: list[tuple[str, str, float]] = []
    for name, target in [
        ("synonyms.json", 1.0),
        ("antonyms.json", 0.5),
        ("unrelated.json", 0.0),
    ]:
        data = json.loads((FIXTURES / name).read_text())
        for a, b in data["pairs"]:
            out.append((a, b, target))
    return out


# --- Embedding backend -----------------------------------------------------


class GloveBackend:
    def __init__(self, name: str):
        self.name = name
        self.model = gensim_api.load(name)
        self.dim = self.model.vector_size

    def has(self, word: str) -> bool:
        return word in self.model

    def encode(self, word: str) -> np.ndarray:
        return self.model[word]

    def similarity(self, a: str, b: str) -> float | None:
        if a not in self.model or b not in self.model:
            return None
        va = self.model[a]
        vb = self.model[b]
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return None
        return float(np.dot(va, vb) / denom)


# --- Strand backend --------------------------------------------------------


class StrandBackend:
    def __init__(self):
        self.codebook = default_codebook()
        self.size = len(self.codebook)

    def has(self, word: str) -> bool:
        return self.codebook.lookup(word) is not None

    def encode(self, word: str):
        return strand_encode(word).strand

    def similarity(self, a: str, b: str) -> float | None:
        from strands import compare

        return compare(a, b).score


# --- Measurements ----------------------------------------------------------


def time_encode(backend, words: list[str], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for w in words:
            backend.encode(w)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best / len(words)


def correlation(backend, pairs: list[tuple[str, str, float]]) -> tuple[float, int, int]:
    targets: list[float] = []
    scores: list[float] = []
    skipped = 0
    for a, b, target in pairs:
        s = backend.similarity(a, b)
        if s is None:
            skipped += 1
            continue
        targets.append(target)
        scores.append(s)
    rho, _ = spearmanr(targets, scores)
    return float(rho), len(targets), skipped


# --- Main ------------------------------------------------------------------


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f}MB"
    return f"{n / 1024 ** 3:.2f}GB"


def main() -> None:
    rng = random.Random(0)
    pairs = load_fixture_pairs()
    print(f"Loaded {len(pairs)} pairs from fixtures")

    # Pick a representative encoding workload — 1000 random unique words from
    # the pair set, repeated.
    word_pool = list({w for a, b, _ in pairs for w in (a, b)})
    rng.shuffle(word_pool)
    encode_words = word_pool[:200]

    print("\nLoading backends...")
    strand = StrandBackend()
    glove50 = GloveBackend("glove-wiki-gigaword-50")
    glove300 = GloveBackend("glove-wiki-gigaword-300")

    # --- Storage ----------------------------------------------------------
    print("\n=== Storage ===")
    print(f"{'Backend':24} {'vocab':>10} {'bytes/word':>12} {'1M-doc index*':>16}")

    for name, vocab, bpw in [
        ("Strand (codon+shade)", strand.size, 4),
        (f"GloVe-50d (float32)", len(glove50.model), glove50.dim * 4),
        (f"GloVe-300d (float32)", len(glove300.model), glove300.dim * 4),
        ("OpenAI 3-small (1536d)", "—", 1536 * 4),
    ]:
        v = f"{vocab:,}" if isinstance(vocab, int) else vocab
        # 1M docs × ~12 tokens/doc avg
        idx_bytes = bpw * 12 * 1_000_000
        print(f"{name:24} {v:>10} {bpw:>12,} {fmt_bytes(idx_bytes):>16}")
    print("  * 1M documents @ 12 tokens/doc average.")

    # --- Encoding speed ---------------------------------------------------
    print("\n=== Encoding speed (warm cache) ===")
    print(f"{'Backend':24} {'µs/word':>12} {'words/sec':>14}")
    for name, b in [
        ("Strand", strand),
        ("GloVe-50d", glove50),
        ("GloVe-300d", glove300),
    ]:
        avg_s = time_encode(b, encode_words, repeat=5)
        print(f"{name:24} {avg_s * 1e6:>12,.1f} {1 / avg_s:>14,.0f}")

    # --- Quality (Spearman ρ vs synonym/antonym/unrelated targets) --------
    print("\n=== Quality — Spearman ρ on combined fixtures ===")
    print(f"{'Backend':24} {'ρ':>8} {'pairs':>8} {'skipped':>10}")
    for name, b in [
        ("Strand", strand),
        ("GloVe-50d", glove50),
        ("GloVe-300d", glove300),
    ]:
        rho, used, skipped = correlation(b, pairs)
        print(f"{name:24} {rho:>8.4f} {used:>8} {skipped:>10}")

    # --- Quality on each fixture separately ------------------------------
    print("\n=== Mean similarity per category (sanity) ===")
    fixtures = {
        "synonyms": json.loads((FIXTURES / "synonyms.json").read_text())["pairs"],
        "antonyms": json.loads((FIXTURES / "antonyms.json").read_text())["pairs"],
        "unrelated": json.loads((FIXTURES / "unrelated.json").read_text())["pairs"],
    }
    print(f"{'Backend':24} {'syn':>8} {'ant':>8} {'unrel':>8}")
    for name, b in [("Strand", strand), ("GloVe-50d", glove50), ("GloVe-300d", glove300)]:
        means = {}
        for cat, plist in fixtures.items():
            vals = []
            for a, b_word in plist:
                s = b.similarity(a, b_word)
                if s is not None:
                    vals.append(s)
            means[cat] = float(np.mean(vals)) if vals else float("nan")
        print(f"{name:24} {means['synonyms']:>8.3f} {means['antonyms']:>8.3f} {means['unrelated']:>8.3f}")


if __name__ == "__main__":
    main()
