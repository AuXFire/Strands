"""Word-sense disambiguation (G3.2 — spec §11.4 advanced features).

Simplified Lesk algorithm: for each polysemous token, pick the synset
whose definition (and example sentences) shares the most words with the
surrounding context.

Used by the encoder to populate ``CodonEntry.sense_rank``: which alt
codon (0 = primary) the token resolved to. This stops C1 multi-sense
match-making at compare time from inflating false positives — the encoder
has already committed to the right sense.

Lesk is fast (no model inference, just set intersection over WordNet
glosses) and deterministic.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None  # type: ignore


_TOK_RE = re.compile(r"[A-Za-z']+")
_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "for", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "not", "this", "that", "it", "its",
})


@lru_cache(maxsize=8192)
def _gloss_words(synset_name: str) -> frozenset[str]:
    """Lowercase content words from a synset's definition + examples."""
    if wn is None:
        return frozenset()
    try:
        s = wn.synset(synset_name)
    except Exception:
        return frozenset()
    text = s.definition() + " " + " ".join(s.examples())
    return frozenset(t.lower() for t in _TOK_RE.findall(text) if t.lower() not in _STOP)


def lesk_select(
    target: str,
    context: list[str],
    *,
    max_senses: int = 8,
) -> int | None:
    """Pick the index of the best WordNet sense for ``target`` given
    ``context`` words. Returns 0..N-1 or None if no synsets exist.
    Index 0 is the primary (most frequent) sense; higher = alternates."""
    if wn is None:
        return None
    synsets = wn.synsets(target.lower())
    if not synsets:
        return None
    if len(synsets) == 1:
        return 0

    ctx_set = frozenset(c.lower() for c in context if c.lower() not in _STOP)
    if not ctx_set:
        return 0  # no context — keep primary sense

    best_idx = 0
    best_score = -1
    for idx, syn in enumerate(synsets[:max_senses]):
        gloss = _gloss_words(syn.name())
        # Score: number of overlapping content words.
        # Tiebreak by sense frequency (lower idx wins).
        score = len(gloss & ctx_set) * 100 - idx
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx
