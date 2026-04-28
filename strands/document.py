"""Document-level strand summarization (spec §11.4).

For long inputs, the per-token strand becomes unwieldy. This module
produces a compact "fingerprint" suitable for corpus-scale retrieval:

  - top-K most informative codons (by TF-IDF-style weight)
  - domain histogram (count per domain code)
  - source size, byte-size, and codon count metadata

Two documents can be compared via fingerprint similarity:
  - cosine over domain histograms (cheap, broad-topic match)
  - Jaccard over top-K codon sets (sharper, content match)
  - blended score combining both
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from strands.codon import DOMAIN_CODES, Codon
from strands.encoder import encode
from strands.strand import Strand


_CODE_DOMAINS = frozenset(c for c in ("CF", "DS", "TS", "OP", "IO", "ER",
                                       "PT", "MD", "TE", "AP", "IN"))


@dataclass(slots=True)
class DocumentFingerprint:
    top_codons: list[tuple[str, int]]      # [(codon_str, count), ...]
    domain_histogram: dict[str, int]       # domain_code -> count
    total_codons: int
    byte_size: int                         # full strand byte size
    fingerprint_bytes: int                 # fingerprint storage cost

    @classmethod
    def from_strand(
        cls,
        strand: Strand,
        *,
        top_k: int = 16,
    ) -> DocumentFingerprint:
        """Compute a fingerprint from an encoded strand."""
        codon_counts: Counter = Counter()
        domain_hist: Counter = Counter()
        for entry in strand.codons:
            codon_str = entry.codon.to_str()
            codon_counts[codon_str] += 1
            domain_hist[entry.codon.domain_code] += 1

        top = codon_counts.most_common(top_k)
        # storage: 5 bytes per top codon (3 codon + 2 count) +
        #          3 bytes per domain bucket (1 code-id + 2 count)
        fp_bytes = len(top) * 5 + len(domain_hist) * 3
        return cls(
            top_codons=top,
            domain_histogram=dict(domain_hist),
            total_codons=len(strand.codons),
            byte_size=strand.byte_size,
            fingerprint_bytes=fp_bytes,
        )

    @classmethod
    def from_text(cls, text: str, *, top_k: int = 16) -> DocumentFingerprint:
        return cls.from_strand(encode(text).strand, top_k=top_k)


def histogram_cosine(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    import math
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def topcodon_jaccard(a: list[tuple[str, int]], b: list[tuple[str, int]]) -> float:
    sa = {c for c, _ in a}
    sb = {c for c, _ in b}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fingerprint_similarity(
    a: DocumentFingerprint,
    b: DocumentFingerprint,
    *,
    histogram_weight: float = 0.5,
) -> float:
    """Blended similarity. Cosine over domain histogram + Jaccard over top
    codons. ``histogram_weight`` controls the mix."""
    h = histogram_cosine(a.domain_histogram, b.domain_histogram)
    j = topcodon_jaccard(a.top_codons, b.top_codons)
    return histogram_weight * h + (1.0 - histogram_weight) * j
