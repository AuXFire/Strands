"""Strand alignment / comparison — pure in-strand byte math.

Two strands compared on a Raspberry Pi with no codebook, no runtime
model, no sidecar files produce a meaningful similarity score from
their bytes alone. The comparator does codon-hierarchy arithmetic plus
in-strand cross-codon relatedness using the two related-codon slots
each token carries (stamped at encode time from ConceptNet 5.7).

Scoring tiers:
  Tier 1 — Codon hierarchy (spec §8.1):
    same domain                       0.25
    + same category                   0.15
    + same concept                    0.35
    + shade similarity (concept)      0.25 × shade_sim
    + shade similarity (partial)      0.05 × shade_sim
  Tier 2 — In-strand related-codon match:
    a's primary == b.related[i]       weight × 0.85 / 255
    a.related[i] == b's primary       weight × 0.85 / 255
    a.related[i] == b.related[j]      min(weights) × 0.55 / 255

  The final per-pair score is max(Tier 1, Tier 2). Tier 1 wins on
  synonyms; Tier 2 fills cross-domain relatedness without external
  data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.codon import Codon
from strands.shade import shade_similarity
from strands.strand import CodonEntry, Strand


# Code-domain IDs for spec §8.3 rule 1 (structural codons get 1.5×).
_CODE_DOMAIN_IDS: frozenset[int] = frozenset({
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A,
})


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


def _hierarchy_score(
    a: Codon, b: Codon, shade_a: int, shade_b: int,
    *, code_weight: bool = False,
) -> float:
    """Tier 1: spec §8.1 codon hierarchy with shade tiebreaker."""
    if a.domain != b.domain:
        return 0.0

    sh_sim = shade_similarity(shade_a, shade_b)
    score = 0.25
    if a.category == b.category:
        score += 0.15
        if a.concept == b.concept:
            score += 0.35
            score += sh_sim * 0.25
        else:
            score += sh_sim * 0.05
    else:
        score += sh_sim * 0.05

    if code_weight and a.domain in _CODE_DOMAIN_IDS:
        score = min(1.0, score * 1.5)
    return score


def _relation_score(a: CodonEntry, b: CodonEntry) -> float:
    """Tier 2: exact-codon match between either token's primary and the
    other's stamped related codons. Reads ONLY the bytes already in the
    two CodonEntry's — no codebook, no graph file, no model.
    """
    best = 0.0

    # a's related ↔ b's primary
    for rel_codon, rel_w in a.related:
        if rel_codon.is_null:
            continue
        if rel_codon == b.codon:
            s = (rel_w / 255.0) * 0.85
            if s > best:
                best = s

    # b's related ↔ a's primary
    for rel_codon, rel_w in b.related:
        if rel_codon.is_null:
            continue
        if rel_codon == a.codon:
            s = (rel_w / 255.0) * 0.85
            if s > best:
                best = s

    # a.related ∩ b.related (second-order — both tokens point at the
    # same neighbor concept)
    for ra, wa in a.related:
        if ra.is_null:
            continue
        for rb, wb in b.related:
            if rb.is_null:
                continue
            if ra == rb:
                s = (min(wa, wb) / 255.0) * 0.55
                if s > best:
                    best = s

    return best


def _pair_score(a: CodonEntry, b: CodonEntry, *, code_aware: bool = False) -> float:
    """Combine Tier 1 (hierarchy) and Tier 2 (in-strand relations).

    Pure byte arithmetic — no external state."""
    h = _hierarchy_score(a.codon, b.codon, a.shade, b.shade, code_weight=code_aware)
    if h >= 0.75:
        # Concept-level codon match — strand has spoken. Skip Tier 2.
        return h
    r = _relation_score(a, b)
    return max(h, r)


def compare_strands(
    strand_a: Strand,
    strand_b: Strand,
    *,
    code_aware: bool = False,
    pattern_bonus: float = 0.0,
) -> ComparisonResult:
    """Greedy alignment with in-strand relatedness scoring.

    Pure byte math — needs only the two strands as input."""
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
            s = _pair_score(codon_a, codon_b, code_aware=code_aware)
            # Spec §8.3 rule 3: position-proximity bonus for code.
            if code_aware and s > 0:
                len_a = max(1, len(strand_a.codons))
                len_b = max(1, len(strand_b.codons))
                pos_diff = abs(i / len_a - j / len_b)
                s += 0.05 * (1.0 - pos_diff)
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

    if pattern_bonus > 0:
        overall = min(1.0, overall + pattern_bonus)

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
