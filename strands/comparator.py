"""Strand alignment / comparison using only data carried by strands."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.adapters import BASE_RELATION_SCALE, ScoringProfile, get_scoring_profile
from strands.codon import Codon
from strands.relations import RelationType
from strands.shade import shade_similarity
from strands.strand import CodonEntry, Strand


_CODE_DOMAIN_IDS: frozenset[int] = frozenset({
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A,
})

def _is_code_heavy(strand: Strand) -> bool:
    if not strand.codons:
        return False
    code_count = sum(1 for e in strand.codons if e.codon.domain in _CODE_DOMAIN_IDS)
    return code_count / len(strand.codons) >= 0.20


def _select_profile(
    strand_a: Strand,
    strand_b: Strand,
    profile: str,
    *,
    conceptnet_bridge: bool,
    code_aware: bool,
) -> ScoringProfile:
    if profile == "default":
        return ScoringProfile(relation_scale=BASE_RELATION_SCALE)
    if profile in {"strict", "topical", "sentence", "code_search"}:
        return get_scoring_profile(profile)
    if not conceptnet_bridge:
        return ScoringProfile(relation_scale=BASE_RELATION_SCALE)
    if code_aware or _is_code_heavy(strand_b) or len(strand_b.codons) >= max(12, len(strand_a.codons) * 3):
        return get_scoring_profile("code_search")
    if len(strand_a.codons) == 1 and len(strand_b.codons) == 1:
        return get_scoring_profile("topical")
    return get_scoring_profile("sentence")


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
                f" <-> {m.b.word or m.b.codon.to_str()} ({m.b.codon.to_str()}:{m.b.shade:02X})"
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
    """Tier 1: codon hierarchy with shade tiebreaker."""
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


def _relation_score(a: CodonEntry, b: CodonEntry, profile: ScoringProfile) -> float:
    """Tier 2: typed in-strand relation matching.

    Positive relation types contribute according to their native label.
    Antonym/opposition edges are handled by a separate contradiction
    penalty so they can reduce an otherwise high hierarchy score.
    """
    best = 0.0

    for rel in a.related:
        if rel.codon.is_null or rel.relation == RelationType.ANTONYM:
            continue
        rel_scale = profile.relation_scale.get(rel.relation, 0.72)
        if rel.codon == b.codon:
            s = (rel.clamped_weight() / 255.0) * rel_scale * profile.relation_multiplier
            best = max(best, s)
        elif profile.relation_hierarchy_weight > 0:
            soft = _hierarchy_score(rel.codon, b.codon, a.shade, b.shade)
            if soft > 0:
                s = (
                    (rel.clamped_weight() / 255.0)
                    * rel_scale
                    * soft
                    * profile.relation_hierarchy_weight
                    * profile.relation_multiplier
                )
                best = max(best, s)

    for rel in b.related:
        if rel.codon.is_null or rel.relation == RelationType.ANTONYM:
            continue
        rel_scale = profile.relation_scale.get(rel.relation, 0.72)
        if rel.codon == a.codon:
            s = (rel.clamped_weight() / 255.0) * rel_scale * profile.relation_multiplier
            best = max(best, s)
        elif profile.relation_hierarchy_weight > 0:
            soft = _hierarchy_score(rel.codon, a.codon, b.shade, a.shade)
            if soft > 0:
                s = (
                    (rel.clamped_weight() / 255.0)
                    * rel_scale
                    * soft
                    * profile.relation_hierarchy_weight
                    * profile.relation_multiplier
                )
                best = max(best, s)

    for rel_a in a.related:
        if rel_a.codon.is_null or rel_a.relation == RelationType.ANTONYM:
            continue
        for rel_b in b.related:
            if rel_b.codon.is_null or rel_b.relation == RelationType.ANTONYM:
                continue
            if rel_a.codon == rel_b.codon:
                type_scale = min(
                    profile.relation_scale.get(rel_a.relation, 0.72),
                    profile.relation_scale.get(rel_b.relation, 0.72),
                )
                s = (
                    min(rel_a.clamped_weight(), rel_b.clamped_weight())
                    / 255.0
                    * type_scale
                    * 0.65
                    * profile.relation_multiplier
                )
                best = max(best, s)

    return best


def _antonym_penalty(a: CodonEntry, b: CodonEntry, profile: ScoringProfile) -> float:
    best = 0.0
    for rel in a.related:
        if rel.relation == RelationType.ANTONYM and rel.codon == b.codon:
            best = max(best, (rel.clamped_weight() / 255.0) * profile.antonym_penalty)
    for rel in b.related:
        if rel.relation == RelationType.ANTONYM and rel.codon == a.codon:
            best = max(best, (rel.clamped_weight() / 255.0) * profile.antonym_penalty)
    return best


def _pair_score(
    a: CodonEntry,
    b: CodonEntry,
    *,
    code_aware: bool = False,
    profile: ScoringProfile,
) -> float:
    """Combine codon hierarchy and in-strand typed relations."""
    h = _hierarchy_score(a.codon, b.codon, a.shade, b.shade, code_weight=code_aware)
    if h >= 0.75:
        return max(0.0, h - _antonym_penalty(a, b, profile))
    r = _relation_score(a, b, profile)
    return max(0.0, max(h, r) - _antonym_penalty(a, b, profile))


def _lexical_coverage(strand_a: Strand, strand_b: Strand) -> float:
    query_words = {e.word.lower() for e in strand_a.codons if e.word}
    target_words = {e.word.lower() for e in strand_b.codons if e.word}
    if not query_words or not target_words:
        return 0.0
    return len(query_words & target_words) / len(query_words)


def _lexical_dice(strand_a: Strand, strand_b: Strand) -> float:
    words_a = {e.word.lower() for e in strand_a.codons if e.word}
    words_b = {e.word.lower() for e in strand_b.codons if e.word}
    if not words_a or not words_b:
        return 0.0
    return (2.0 * len(words_a & words_b)) / (len(words_a) + len(words_b))


def compare_strands(
    strand_a: Strand,
    strand_b: Strand,
    *,
    code_aware: bool = False,
    conceptnet_bridge: bool = False,
    profile: str = "auto",
    pattern_bonus: float = 0.0,
) -> ComparisonResult:
    """Greedy alignment with typed in-strand relatedness scoring."""
    scoring = _select_profile(
        strand_a, strand_b,
        profile,
        conceptnet_bridge=conceptnet_bridge,
        code_aware=code_aware,
    )
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
            s = _pair_score(codon_a, codon_b, code_aware=code_aware, profile=scoring)
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
    if scoring.query_coverage and strand_a.codons:
        if scoring.symmetric_coverage:
            denom = len(strand_a.codons) + len(strand_b.codons)
            overall = min(1.0, (2.0 * total_score) / denom) if denom > 0 else 0.0
        else:
            overall = min(1.0, total_score / len(strand_a.codons))
        if scoring.lexical_weight > 0:
            lexical = (
                _lexical_dice(strand_a, strand_b)
                if scoring.symmetric_coverage
                else _lexical_coverage(strand_a, strand_b)
            )
            semantic_weight = 1.0 - scoring.lexical_weight
            overall = (semantic_weight * overall) + (scoring.lexical_weight * lexical)
    else:
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
