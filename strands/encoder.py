"""Text encoder pipeline: text → tokenize → lemmatize → codebook lookup → shade → Strand."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.codebook import Codebook, default_codebook
from strands.lemmatizer import lemmatize
from strands.shade import compute_shade
from strands.strand import CodonEntry, Strand
from strands.tokenizer import tokenize


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


def encode(text: str, *, codebook: Codebook | None = None) -> EncodeResult:
    cb = codebook or default_codebook()

    raw_tokens = tokenize(text)
    entries: list[CodonEntry] = []
    unknowns: list[str] = []

    for token in raw_tokens:
        # Try direct lookup first (handles short common words like "go").
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
