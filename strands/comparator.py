"""Strand alignment / comparison (spec §8)."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.shade import shade_similarity
from strands.strand import CodonEntry, Strand


@dataclass(frozen=True, slots=True)
class Match:
    a: CodonEntry
    b: CodonEntry
    score: float


@dataclass(slots=True)
class ComparisonResult:
    score: float
    matches: list[Match] = field(default_factory=list)
    unmatched_a: list[CodonEntry] = field(default_factory=list)
    unmatched_b: list[CodonEntry] = field(default_factory=list)

    def explain(self) -> str:
        lines: list[str] = [f"Overall score: {self.score:.4f}"]
        for m in self.matches:
            lines.append(
                f"  {m.a.word or m.a.codon.to_str()} ({m.a.codon.to_str()}:{m.a.shade:02X})"
                f" ↔ {m.b.word or m.b.codon.to_str()} ({m.b.codon.to_str()}:{m.b.shade:02X})"
                f"  score={m.score:.3f}"
            )
        if self.unmatched_a:
            lines.append(
                "  unmatched (A): "
                + ", ".join(e.word or e.codon.to_str() for e in self.unmatched_a)
            )
        if self.unmatched_b:
            lines.append(
                "  unmatched (B): "
                + ", ".join(e.word or e.codon.to_str() for e in self.unmatched_b)
            )
        return "\n".join(lines)


def _pair_score(a: CodonEntry, b: CodonEntry) -> float:
    if a.codon.domain != b.codon.domain:
        return 0.0
    score = 0.25
    if a.codon.category == b.codon.category:
        score += 0.15
        if a.codon.concept == b.codon.concept:
            score += 0.35
            score += shade_similarity(a.shade, b.shade) * 0.25
    return score


def compare_strands(strand_a: Strand, strand_b: Strand) -> ComparisonResult:
    matches: list[Match] = []
    used_b: set[int] = set()
    matched_a_idx: set[int] = set()
    total_score = 0.0

    for i, codon_a in enumerate(strand_a.codons):
        best_j: int | None = None
        best_score = 0.0
        best_codon_b: CodonEntry | None = None

        for j, codon_b in enumerate(strand_b.codons):
            if j in used_b:
                continue
            s = _pair_score(codon_a, codon_b)
            if s > best_score:
                best_score = s
                best_j = j
                best_codon_b = codon_b

        if best_j is not None and best_score > 0 and best_codon_b is not None:
            used_b.add(best_j)
            matched_a_idx.add(i)
            matches.append(Match(a=codon_a, b=best_codon_b, score=best_score))
            total_score += best_score

    max_len = max(len(strand_a.codons), len(strand_b.codons))
    overall = total_score / max_len if max_len > 0 else 0.0

    unmatched_a = [
        codon for i, codon in enumerate(strand_a.codons) if i not in matched_a_idx
    ]
    unmatched_b = [
        codon for j, codon in enumerate(strand_b.codons) if j not in used_b
    ]

    return ComparisonResult(
        score=overall,
        matches=matches,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
    )
