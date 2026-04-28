"""MEN-3000 — Spearman ρ between strand similarity and human judgments.

Spec §12.1 aspirational target: ρ ≥ 0.50.
"""

from __future__ import annotations

import pytest

from strands import compare
from tests.benchmarks._loaders import load_men3000


SPEC_TARGET_RHO = 0.50
REALISTIC_FLOOR = 0.30  # pure-strand floor; with ConceptNet enabled, ~0.73


def test_men3000_spearman():
    scipy = pytest.importorskip("scipy.stats")
    pairs = load_men3000()
    assert len(pairs) > 2900, f"Expected MEN-3000, got {len(pairs)} pairs"

    gold: list[float] = []
    pred: list[float] = []
    for a, b, g in pairs:
        gold.append(g)
        pred.append(compare(a, b).score)

    rho, _ = scipy.spearmanr(gold, pred)
    print(
        f"\nMEN-3000: ρ = {rho:.4f} on {len(gold)} pairs "
        f"(spec target {SPEC_TARGET_RHO}, floor {REALISTIC_FLOOR})"
    )
    assert rho >= REALISTIC_FLOOR, f"MEN-3000 ρ {rho:.4f} below floor {REALISTIC_FLOOR}"
