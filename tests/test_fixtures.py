"""Run synonym/antonym/unrelated fixture pairs against the comparator.

These follow spec §12.4 expectations. Targets are slightly looser than the
spec because Phase 1's codebook is intentionally smaller (~13k entries).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands import compare

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_synonyms_pass_rate():
    fx = _load("synonyms.json")
    threshold = fx["min_score"]
    passes = 0
    failures: list[tuple[str, str, float]] = []
    for a, b in fx["pairs"]:
        score = compare(a, b).score
        if score >= threshold:
            passes += 1
        else:
            failures.append((a, b, score))
    rate = passes / len(fx["pairs"])
    # MVP target: at least 80% of synonym pairs hit the threshold.
    assert rate >= 0.80, (
        f"Synonym pass rate {rate:.2%} below 80% target. "
        f"First failures: {failures[:5]}"
    )


def test_antonyms_in_band():
    fx = _load("antonyms.json")
    lo, hi = fx["min_score"], fx["max_score"]
    in_band = 0
    failures: list[tuple[str, str, float]] = []
    for a, b in fx["pairs"]:
        score = compare(a, b).score
        if lo <= score <= hi:
            in_band += 1
        else:
            failures.append((a, b, score))
    rate = in_band / len(fx["pairs"])
    # Antonyms should land in the same domain but typically different concept.
    assert rate >= 0.75, (
        f"Antonym in-band rate {rate:.2%} below 75% target. "
        f"First failures: {failures[:5]}"
    )


def test_unrelated_score_low():
    fx = _load("unrelated.json")
    threshold = fx["max_score"]
    passes = 0
    failures: list[tuple[str, str, float]] = []
    for a, b in fx["pairs"]:
        score = compare(a, b).score
        if score <= threshold:
            passes += 1
        else:
            failures.append((a, b, score))
    rate = passes / len(fx["pairs"])
    assert rate >= 0.85, (
        f"Unrelated low-score rate {rate:.2%} below 85% target. "
        f"First failures: {failures[:5]}"
    )


@pytest.mark.parametrize("a,b,expected_min", [
    ("happy", "joyful", 0.80),
    ("happy dog", "joyful puppy", 0.80),
    ("car", "automobile", 0.80),
    ("dog", "puppy", 0.80),
])
def test_handpicked_synonyms(a: str, b: str, expected_min: float):
    score = compare(a, b).score
    assert score >= expected_min, f"{a} ↔ {b}: {score:.3f} < {expected_min}"
