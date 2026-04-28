"""CodeXGLUE WebQuery NL→Code retrieval benchmark.

For each NL query, rank all candidate code snippets by similarity. Report
MRR (Mean Reciprocal Rank) of the gold code in the ranked list.

Spec §12.3 target for code search: MRR >= 0.25.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands import compare_strands, encode, encode_code

DATA = Path(__file__).parent / "data" / "codesearch_webquery.json"

REALISTIC_FLOOR = 0.30  # spec target 0.25; achieved ~0.35 on first 100 subset


def _load(limit: int = 200) -> list[dict]:
    """Load up to ``limit`` records. Subsampled for test speed."""
    data = json.loads(DATA.read_text())
    return data[:limit]


def _detect_lang(code: str) -> str:
    if "def " in code or "import " in code:
        return "python"
    if "function " in code or "const " in code or "let " in code:
        return "javascript"
    if "fn " in code or "let mut" in code:
        return "rust"
    return "python"


@pytest.mark.slow
def test_codesearch_mrr():
    records = _load(limit=100)  # tighter subset for CI speed

    # Encode every code snippet once
    code_strands = []
    for r in records:
        try:
            s = encode_code(r["code"], language=_detect_lang(r["code"])).strand
        except Exception:
            s = encode(r["code"]).strand
        code_strands.append(s)

    # Rank gold code in the ranked candidate list per query.
    # Use ConceptNet bridge — NL queries are sentence-like and benefit
    # from the same sentence-mode that wins on STS/SICK.
    rrs = []
    for i, r in enumerate(records):
        q_strand = encode(r["doc"]).strand
        scores = []
        for j, c_strand in enumerate(code_strands):
            sc = compare_strands(q_strand, c_strand,
                                 conceptnet_bridge=True).score
            scores.append((sc, j))
        scores.sort(reverse=True)
        # Find rank of the gold (j == i)
        for rank, (_, j) in enumerate(scores, start=1):
            if j == i:
                rrs.append(1.0 / rank)
                break

    mrr = sum(rrs) / len(rrs)
    print(f"\nCodeXGLUE WebQuery (n={len(records)}): MRR = {mrr:.4f}")
    assert mrr >= REALISTIC_FLOOR, f"MRR {mrr:.4f} below floor {REALISTIC_FLOOR}"
