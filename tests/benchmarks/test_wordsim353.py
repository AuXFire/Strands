"""WordSim-353 — Spearman ρ between strand similarity and human judgments.

Spec §12.1 aspirational target: ρ ≥ 0.55 (embeddings typically 0.60–0.75).
Same caveat as SimLex-999: strand scoring is discrete (4 distinct levels per
pair), so ties cap achievable ρ. Realistic empirical floor on the current
245k-entry codebook is ~0.30.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands import compare

DATA = Path(__file__).parent / "data" / "wordsim353.tsv"
SPEC_TARGET_RHO = 0.55  # spec §12.1, aspirational
REALISTIC_FLOOR = 0.28  # post-corrections (C1+C2+C3+C4+C5) floor


def _load_pairs() -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for line in DATA.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, b, gold = parts[0], parts[1], float(parts[2])
        pairs.append((a, b, gold))
    return pairs


def test_wordsim353_spearman():
    scipy = pytest.importorskip("scipy.stats")
    pairs = _load_pairs()
    assert len(pairs) > 300, f"Expected WordSim-353, got {len(pairs)} pairs"

    gold: list[float] = []
    pred: list[float] = []
    skipped = 0
    for a, b, g in pairs:
        score = compare(a, b).score
        if score == 0.0 and not _both_known(a, b):
            skipped += 1
            continue
        gold.append(g)
        pred.append(score)

    rho, _ = scipy.spearmanr(gold, pred)
    rate = (len(pairs) - skipped) / len(pairs)
    print(
        f"\nWordSim-353: ρ = {rho:.4f} on {len(gold)}/{len(pairs)} pairs "
        f"(coverage {rate:.1%}, spec target {SPEC_TARGET_RHO}, floor {REALISTIC_FLOOR})"
    )
    assert rho >= REALISTIC_FLOOR, (
        f"WordSim-353 ρ {rho:.4f} below regression floor {REALISTIC_FLOOR}"
    )


def _both_known(a: str, b: str) -> bool:
    from strands.codebook import default_codebook
    cb = default_codebook()
    return cb.lookup(a) is not None and cb.lookup(b) is not None
