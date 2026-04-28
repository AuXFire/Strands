"""Strand alignment / comparison (spec §8 + post-benchmark corrections).

Active corrections:
  C1 — multi-sense max: when the codebook supplies alternative codons for a
       word, take max(score) across all primary/alt combinations.
  C2 — WordNet path-similarity bridge: when the codon-level score is below
       the domain-match floor, fall back to path_similarity between the
       entries' synset names (× 0.4 cap so it never beats a true codon
       match).
  C3 — shade tiebreaker for partial matches: a small shade-similarity bonus
       (× 0.05) on same-domain-only and same-domain+category pairs, in
       addition to the spec §8.1 main bonus (× 0.25) at concept match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from strands.codon import Codon
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


def _codon_pair_score(a: Codon, b: Codon, shade_a: int, shade_b: int) -> float:
    """Pure codon+shade scoring; spec §8.1 plus C3 shade tiebreaker."""
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
    return score


@lru_cache(maxsize=8192)
def _path_similarity(syn_a: str, syn_b: str) -> float | None:
    """Memoised WordNet path similarity. Returns None on any failure."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None
    try:
        s_a = wn.synset(syn_a)
        s_b = wn.synset(syn_b)
        return s_a.path_similarity(s_b)
    except Exception:
        return None


def _pair_score(a: CodonEntry, b: CodonEntry, *, wordnet_bridge: bool = True) -> float:
    """C1 multi-sense + C2 WordNet bridge + C3 shade tiebreaker."""
    a_codons = (a.codon,) + a.alt_codons
    b_codons = (b.codon,) + b.alt_codons

    best = 0.0
    for ca in a_codons:
        for cb in b_codons:
            s = _codon_pair_score(ca, cb, a.shade, b.shade)
            if s > best:
                best = s

    # C2: cross-domain WordNet bridge — only when no codon variant gave any
    # domain-level credit, and only up to 0.4 (less than a category match).
    if best < 0.25 and wordnet_bridge and a.synset and b.synset:
        sim = _path_similarity(a.synset, b.synset)
        if sim is not None and sim > 0:
            best = max(best, sim * 0.4)

    return best


def compare_strands(
    strand_a: Strand,
    strand_b: Strand,
    *,
    wordnet_bridge: bool = True,
) -> ComparisonResult:
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
            s = _pair_score(codon_a, codon_b, wordnet_bridge=wordnet_bridge)
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
