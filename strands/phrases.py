from __future__ import annotations

from strands.adapters import FrameSpec, iter_frame_specs
from strands.strand import CodonEntry


ROLE_FRAME = 5
FEATURE_PHRASE = 0x0001
FEATURE_FRAME = 0x0002


def add_frame_entries(entries: list[CodonEntry]) -> list[CodonEntry]:
    """Append deterministic native frame codons inferred from token words."""
    words = {entry.word.lower() for entry in entries if entry.word}
    if not words:
        return entries

    seen = {entry.codon for entry in entries if entry.features & FEATURE_FRAME}
    out = list(entries)
    for rule in iter_frame_specs():
        if rule.codon in seen or not rule.matches(words):
            continue
        out.append(CodonEntry(
            codon=rule.codon,
            shade=0x56,
            word=rule.name,
            role=rule.role,
            features=rule.feature,
        ))
        seen.add(rule.codon)
    return out
