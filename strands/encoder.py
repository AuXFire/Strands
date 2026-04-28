"""Text encoder pipeline: text → tokenize → lemmatize → codebook lookup →
context-aware shade → Strand v2 with stamped related codons."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.codebook import Codebook, default_codebook
from strands.context_shade import apply_context, scan_context
from strands.lemmatizer import lemmatize
from strands.shade import compute_shade
from strands.strand import VERSION_V2, CodonEntry, Strand
from strands.tokenizer import STOP_WORDS


@dataclass(slots=True)
class EncodeResult:
    strand: Strand
    text: str
    unknowns: list[str] = field(default_factory=list)

    @property
    def byte_size(self) -> int:
        return self.strand.byte_size

    @property
    def strand_text(self) -> str:
        return self.strand.to_text()


def encode(
    text: str,
    *,
    codebook: Codebook | None = None,
    context_aware: bool = True,
) -> EncodeResult:
    """Encode ``text`` into a strand-v2 Strand.

    Each output token carries:
      - Primary codon (3 bytes) and shade (1 byte) — spec §3.
      - Up to two related codons with quantized weights (2 × 4 bytes) —
        stamped from the codebook's ConceptNet-derived ``rel`` field.

    The resulting strand is fully self-contained: comparing two strands
    requires only their bytes — no codebook, no runtime model, no
    sidecar files.
    """
    cb = codebook or default_codebook()

    if context_aware:
        all_tokens, hints = scan_context(text)
    else:
        from strands.tokenizer import tokenize as _tok

        all_tokens = _tok(text, drop_stop_words=False)
        hints = None

    entries: list[CodonEntry] = []
    unknowns: list[str] = []

    for idx, token in enumerate(all_tokens):
        if token in STOP_WORDS:
            continue

        cb_entry = cb.lookup(token)
        if cb_entry is None:
            lemma = lemmatize(token)
            cb_entry = cb.lookup(lemma)
            if cb_entry is None:
                unknowns.append(token)
                continue
            lookup_word = lemma
        else:
            lookup_word = token

        shade = compute_shade(lookup_word, cb_entry.shade_hint)
        if hints is not None:
            shade = apply_context(shade, idx, hints)

        entries.append(
            CodonEntry(
                codon=cb_entry.codon,
                shade=shade,
                word=lookup_word,
                related=cb_entry.related,
                synset=cb_entry.synset,
            )
        )

    return EncodeResult(
        strand=Strand(codons=entries, version=VERSION_V2),
        text=text,
        unknowns=unknowns,
    )
