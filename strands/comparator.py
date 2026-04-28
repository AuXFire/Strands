"""Strand alignment / comparison (spec §8 + post-benchmark corrections).

Active corrections:
  C1 — multi-sense max: take max(score) across all primary/alt combinations
       when the codebook supplies alternative codons for a word.
  C2 — WordNet bridge: when codon-level score is low, use Wu-Palmer +
       path-similarity over the cross-product of synsets (capped at 0.6).
  C3 — shade tiebreaker: small shade-similarity bonus on partial matches.
  C6 — ConceptNet bridge (opt-in): when WordNet bridge is also low, use
       ConceptNet Numberbatch cosine (capped at 0.55). Disabled by default;
       enable via ``conceptnet_bridge=True`` or env var
       ``STRANDS_CONCEPTNET=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from strands.codon import Codon
from strands.relatedness import (
    conceptnet_mean_vector,
    conceptnet_word_similarity,
    wordnet_similarity,
    wordnet_word_similarity,
)
from strands.shade import shade_similarity
from strands.strand import CodonEntry, Strand


_CONCEPTNET_DEFAULT = os.environ.get("STRANDS_CONCEPTNET", "0") == "1"


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


# Code domain IDs (spec §5.2). Used for structural-weight bonus (§8.3).
_CODE_DOMAIN_IDS: frozenset[int] = frozenset({
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A,
})


def _is_code_domain(domain: int) -> bool:
    return domain in _CODE_DOMAIN_IDS


def _codon_pair_score(a: Codon, b: Codon, shade_a: int, shade_b: int,
                      *, code_weight: bool = False) -> float:
    """Spec §8.1 plus C3 shade tiebreaker. Spec §8.3 rule 1 — when
    ``code_weight`` is True, code-domain matches are weighted 1.5x."""
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

    if code_weight and _is_code_domain(a.domain):
        score = min(1.0, score * 1.5)
    return score


def _pair_score(
    a: CodonEntry,
    b: CodonEntry,
    *,
    wordnet_bridge: bool = True,
    conceptnet_bridge: bool = False,
    code_aware: bool = False,
) -> float:
    """C1 multi-sense + C2 WordNet bridge + C3 shade + C6 ConceptNet bridge.

    When ``code_aware`` is True, applies spec §8.3 rule 1 (structural codons
    get 1.5x weight)."""
    a_codons = (a.codon,) + a.alt_codons
    b_codons = (b.codon,) + b.alt_codons

    best = 0.0
    for ca in a_codons:
        for cb in b_codons:
            s = _codon_pair_score(ca, cb, a.shade, b.shade, code_weight=code_aware)
            if s > best:
                best = s

    codon_best = best

    # Tier 3 — ConceptNet/Numberbatch is the most reliable continuous
    # relatedness signal available. When both words are in vocabulary,
    # CN drives the ranking. Strand only contributes a synonym lift for
    # pairs where strand identifies a perfect concept match.
    cn = None
    if conceptnet_bridge and a.word and b.word:
        cn = conceptnet_word_similarity(a.word, b.word)

    if cn is not None and cn > 0:
        # Pure CN ranking. Empirically this gives the strongest correlation
        # across both similarity (SimLex/SimVerb) and relatedness
        # (WordSim/MEN/RG-65) benchmarks. Codon-level signals in this
        # range hurt more than they help (false positives from same-concept
        # collisions like loop/belt, monk/slave).
        return cn

    # Tier 2 — WordNet bridge (only relevant when ConceptNet is unavailable
    # or one of the words is OOV in Numberbatch). A threshold of 0.30
    # filters out the ~0.2 noise floor where any two physical_entity nouns
    # share entity.n.01 as a remote ancestor (soldier/fruit, etc.).
    if wordnet_bridge:
        sim = None
        if a.synset and b.synset:
            sim = wordnet_similarity(a.synset, b.synset)
        if sim is None and a.word and b.word:
            sim = wordnet_word_similarity(a.word, b.word)
        if sim is not None and sim > 0.30:
            best = max(best, (sim - 0.30) / 0.70 * 0.85)

    return best


# Auto-switch to sentence-mode (mean ConceptNet vector cosine) when either
# strand has more than this many content codons. Sentence-mode is a much
# stronger signal than greedy alignment for STS-style benchmarks.
_SENTENCE_MODE_THRESHOLD = 4


def compare_strands(
    strand_a: Strand,
    strand_b: Strand,
    *,
    wordnet_bridge: bool = True,
    conceptnet_bridge: bool | None = None,
    sentence_mode: bool | None = None,
    code_aware: bool = False,
    pattern_bonus: float = 0.0,
) -> ComparisonResult:
    """Align two strands and produce a similarity score.

    When ``sentence_mode`` is True (or auto-detected from strand length),
    ConceptNet mean-vector cosine is used as the primary score. This
    matches the GloVe/Word2Vec sentence-similarity baseline but uses the
    higher-quality ConceptNet Numberbatch vectors.
    """
    if conceptnet_bridge is None:
        conceptnet_bridge = _CONCEPTNET_DEFAULT

    if sentence_mode is None:
        sentence_mode = (
            conceptnet_bridge
            and (
                len(strand_a.codons) > _SENTENCE_MODE_THRESHOLD
                or len(strand_b.codons) > _SENTENCE_MODE_THRESHOLD
            )
        )

    if sentence_mode and conceptnet_bridge:
        from strands.relatedness import conceptnet_mean_vector, vector_cosine

        words_a = [c.word for c in strand_a.codons if c.word]
        words_b = [c.word for c in strand_b.codons if c.word]
        va = conceptnet_mean_vector(words_a)
        vb = conceptnet_mean_vector(words_b)
        if va is not None and vb is not None:
            score = max(0.0, vector_cosine(va, vb))
            return ComparisonResult(score=score, matches=[],
                                    unmatched_a=[], unmatched_b=[])
        # If CN vectors unavailable for either side, fall through to alignment.

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
            s = _pair_score(
                codon_a,
                codon_b,
                wordnet_bridge=wordnet_bridge,
                conceptnet_bridge=conceptnet_bridge,
                code_aware=code_aware,
            )
            # Spec §8.3 rule 3: position-proximity bonus up to 0.05.
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

    # Spec §8.3 rule 2 — pattern bonus when both strands share a detected pattern.
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
