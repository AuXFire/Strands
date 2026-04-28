"""RG-65 (Rubenstein-Goodenough) — small-but-classic similarity benchmark.

Spec §12.1 aspirational target: ρ ≥ 0.70 (embeddings 0.75–0.85).
"""

from __future__ import annotations

import pytest

from strands import compare
from tests.benchmarks._loaders import load_rg65


SPEC_TARGET_RHO = 0.70
REALISTIC_FLOOR = 0.55


def test_rg65_spearman():
    scipy = pytest.importorskip("scipy.stats")
    pairs = load_rg65()
    assert len(pairs) >= 60, f"Expected RG-65, got {len(pairs)} pairs"

    gold: list[float] = []
    pred: list[float] = []
    for a, b, g in pairs:
        gold.append(g)
        pred.append(compare(a, b).score)

    rho, _ = scipy.spearmanr(gold, pred)
    print(
        f"\nRG-65: ρ = {rho:.4f} on {len(gold)} pairs "
        f"(spec target {SPEC_TARGET_RHO}, floor {REALISTIC_FLOOR})"
    )
    assert rho >= REALISTIC_FLOOR, f"RG-65 ρ {rho:.4f} below floor {REALISTIC_FLOOR}"
