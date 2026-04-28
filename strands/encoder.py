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
    wsd: bool = False,
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

    if wsd:
        from strands.wsd import lesk_select
        context_words = [t for t in all_tokens if t not in STOP_WORDS]

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

        sense_rank = 0
        chosen_codon = cb_entry.codon
        chosen_alts = cb_entry.alt_codons
        if wsd and cb_entry.alt_codons:
            sense_idx = lesk_select(lookup_word, context_words)
            if sense_idx is not None and sense_idx > 0:
                # Promote the chosen alt to primary; demote primary into alts.
                all_codons = (cb_entry.codon,) + cb_entry.alt_codons
                if sense_idx < len(all_codons):
                    chosen_codon = all_codons[sense_idx]
                    chosen_alts = tuple(
                        c for i, c in enumerate(all_codons) if i != sense_idx
                    )
                    sense_rank = sense_idx

        entries.append(
            CodonEntry(
                codon=chosen_codon,
                shade=shade,
                word=lookup_word,
                alt_codons=chosen_alts,
                synset=cb_entry.synset,
                sense_rank=sense_rank,
                source_position=idx & 0xFFFF,
            )
        )

    return EncodeResult(strand=Strand(codons=entries), text=text, unknowns=unknowns)
