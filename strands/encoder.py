"""Text encoder pipeline: text → tokenize → lemmatize → codebook lookup →
context-aware shade → Strand."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.codebook import Codebook, default_codebook
from strands.context_shade import apply_context, scan_context
from strands.lemmatizer import lemmatize
from strands.shade import compute_shade
from strands.strand import CodonEntry, Strand
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
    """Encode ``text`` into a Strand.

    When ``context_aware`` is True (default), spec §4 modifiers are applied:
    intensifiers boost intensity bits, negation flips polarity bits, formal
    register markers boost formality bits.
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
                alt_codons=cb_entry.alt_codons,
                synset=cb_entry.synset,
            )
        )

    return EncodeResult(strand=Strand(codons=entries), text=text, unknowns=unknowns)
