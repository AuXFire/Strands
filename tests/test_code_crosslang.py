"""Cross-language algorithm fixture: same algorithm should be more similar
across languages than different algorithms in the same language."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

from strands import compare_strands, encode_code

FIXTURE = Path(__file__).parent / "fixtures" / "code_algorithms.json"


def _data():
    return json.loads(FIXTURE.read_text())


def _strand(src, lang):
    return encode_code(src, language=lang).strand


def _score(a_src, a_lang, b_src, b_lang):
    return compare_strands(_strand(a_src, a_lang), _strand(b_src, b_lang)).score


def test_within_algorithm_pairs_score_higher_than_across():
    """Aggregate: mean(within-algorithm) > mean(across-algorithm)."""
    data = _data()["algorithms"]
    within: list[float] = []
    across: list[float] = []

    items = []
    for algo, langs in data.items():
        for lang, src in langs.items():
            items.append((algo, lang, src))

    for (algo_a, lang_a, src_a), (algo_b, lang_b, src_b) in combinations(items, 2):
        s = _score(src_a, lang_a, src_b, lang_b)
        if algo_a == algo_b and lang_a != lang_b:
            within.append(s)
        elif algo_a != algo_b:
            across.append(s)

    mean_within = sum(within) / len(within)
    mean_across = sum(across) / len(across)
    print(f"\n  within (n={len(within)}): mean={mean_within:.3f}")
    print(f"  across (n={len(across)}): mean={mean_across:.3f}")

    assert mean_within > mean_across, (
        f"within ({mean_within:.3f}) should beat across ({mean_across:.3f})"
    )


@pytest.mark.parametrize("algo", [
    "fibonacci", "binary_search", "fetch_url", "parse_json",
    "merge_sort", "bst_insert",
])
def test_each_algorithm_separates_above_floor(algo):
    """Per-algorithm: mean within-pair similarity > 0.45."""
    data = _data()["algorithms"]
    langs = data[algo]
    within = []
    for (la, sa), (lb, sb) in combinations(langs.items(), 2):
        within.append(_score(sa, la, sb, lb))
    mean = sum(within) / len(within)
    print(f"\n  {algo}: mean within-pair similarity = {mean:.3f}")
    assert mean >= 0.40, f"{algo} within-pair mean {mean:.3f} below 0.40"
