#!/usr/bin/env python
"""Comprehensive benchmark orchestrator.

Each section runs as a subprocess so memory is reclaimed between
benchmarks (Numberbatch ~1.2GB + GloVe-300 ~1.4GB + GloVe-50 ~80MB
together cause GC thrash if held simultaneously through long passes).

Each subprocess emits one JSON record to stdout. We collect them and
print a unified table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).parent

SECTIONS = [
    ("_bench_word.py", "A. Word similarity"),
    ("_bench_sentence.py", "B. Sentence STS"),
    ("_bench_codesearch.py", "C. Code retrieval — CodeXGLUE WebQuery"),
    ("_bench_crosslang.py", "E. Cross-language algorithm fixture"),
    ("_bench_clone.py", "D. Code clone detection — BigCloneBench"),
]


def run_section(script: str) -> dict[str, Any] | None:
    cmd = [sys.executable, str(ROOT / script)]
    try:
        out = subprocess.check_output(cmd, timeout=1800)
    except subprocess.CalledProcessError as e:
        print(f"  ! subprocess failed: {e}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"  ! subprocess timed out after 30 min", flush=True)
        return None
    try:
        return json.loads(out.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"  ! parse error: {e}", flush=True)
        print(out.decode("utf-8")[:500])
        return None


def fmt_row(name: str, n: int, strand: float, baseline: float) -> str:
    diff = strand - baseline
    verdict = "WIN" if diff > 0.01 else "LOSE" if diff < -0.01 else "TIE"
    return (
        f"{name:<28} {n:>5} | {strand:>8.4f}  {baseline:>10.4f} | "
        f"{verdict:>4} ({diff:+.3f})"
    )


def main() -> None:
    sections: list[dict[str, Any]] = []

    for script, _ in SECTIONS:
        print(f"Running {script} …", flush=True)
        result = run_section(script)
        if result is not None:
            sections.append(result)
            print(f"  done", flush=True)

    print()
    print("=" * 90)
    print("STRAND vs GLOVE-300 — comprehensive benchmark")
    print("=" * 90)

    all_results: list[tuple[str, float, float, str]] = []
    for section in sections:
        title = section["section"]
        metric = section["metric"]
        print(f"\n--- {title}  ({metric}) ---")
        print(f"{'Dataset':<28} {'pairs':>5} | {'Strand':>8}  {'GloVe-300':>10} | verdict")
        print("-" * 78)
        for row in section["rows"]:
            print(fmt_row(row["name"], row["n"], row["strand"], row["glove300"]))
            verdict = "WIN" if row["strand"] - row["glove300"] > 0.01 else (
                "LOSE" if row["strand"] - row["glove300"] < -0.01 else "TIE"
            )
            all_results.append((row["name"], row["strand"], row["glove300"], verdict))

    print()
    print("=" * 90)
    wins = sum(1 for r in all_results if r[3] == "WIN")
    losses = sum(1 for r in all_results if r[3] == "LOSE")
    ties = sum(1 for r in all_results if r[3] == "TIE")
    print(f"TOTAL: {len(all_results)} benchmarks  |  WINS = {wins}  TIES = {ties}  LOSSES = {losses}")
    if all_results:
        avg_s = mean(r[1] for r in all_results)
        avg_g = mean(r[2] for r in all_results)
        print(f"Average score:  strand = {avg_s:.4f}   GloVe-300 = {avg_g:.4f}   diff = {avg_s - avg_g:+.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
