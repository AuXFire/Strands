"""Context-aware shade modification (spec §4).

Scans the token stream for modifiers that adjust an adjacent content word's
shade bits:

  - Intensifiers ("very", "extremely") → boost intensity bits.
  - Diminishers ("slightly", "somewhat") → reduce intensity bits.
  - Negation ("not", "never") → flip polarity bits.
  - Formal-register markers ("hereby", "wherein") → boost all subsequent
    words' formality bits within the same sentence.

Operates on the raw token stream (before stop-word filtering) so modifier
words like "not" and "very" — which are stop words — are still seen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INTENSIFIERS: dict[str, int] = {
    "extremely": 3, "overwhelmingly": 3, "devastatingly": 3,
    "incredibly": 3, "exceptionally": 3, "tremendously": 3,
    "very": 2, "highly": 2, "deeply": 2, "greatly": 2, "really": 2,
    "quite": 2, "remarkably": 2, "particularly": 2,
    "fairly": 1, "rather": 1, "moderately": 1, "reasonably": 1,
    "somewhat": 0, "slightly": 0, "mildly": 0, "barely": 0, "hardly": 0,
    "scarcely": 0, "marginally": 0,
}

NEGATIONS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "without", "n't",
})

FORMAL_MARKERS: frozenset[str] = frozenset({
    "hereby", "thereof", "wherein", "whereas", "heretofore", "hereinafter",
    "aforementioned", "thus", "thereby", "hence", "moreover", "furthermore",
    "nevertheless", "notwithstanding", "pursuant", "accordingly",
    "henceforth", "respectively",
})

_TOK_RE = re.compile(r"[A-Za-z']+|[.!?]")
_SENTENCE_BREAK = frozenset({".", "!", "?"})


@dataclass(slots=True)
class ContextHints:
    intensity_override: dict[int, int]  # token-index -> new intensity bits (0-3)
    polarity_flip: set[int]              # token-indices to flip polarity
    formality_boost: dict[int, int]      # token-index -> formality delta (0-3)


def scan_context(text: str) -> tuple[list[str], ContextHints]:
    """Tokenize and scan for modifiers. Returns (tokens, hints) where
    ``tokens`` are lowercase letters/apostrophe tokens (sentence-break
    punctuation included as separators but NOT in the returned list)."""
    raw = _TOK_RE.findall(text.lower())
    tokens: list[str] = []
    intensity_override: dict[int, int] = {}
    polarity_flip: set[int] = set()
    formality_boost: dict[int, int] = {}

    pending_intensity: int | None = None
    pending_negation = False
    formal_register_active = False

    for tok in raw:
        if tok in _SENTENCE_BREAK:
            pending_intensity = None
            pending_negation = False
            formal_register_active = False
            continue

        if tok in INTENSIFIERS:
            pending_intensity = INTENSIFIERS[tok]
            continue
        if tok in NEGATIONS:
            pending_negation = True
            continue
        if tok in FORMAL_MARKERS:
            formal_register_active = True

        idx = len(tokens)
        tokens.append(tok)

        if pending_intensity is not None:
            intensity_override[idx] = pending_intensity
            pending_intensity = None
        if pending_negation:
            polarity_flip.add(idx)
            pending_negation = False
        if formal_register_active:
            formality_boost[idx] = max(formality_boost.get(idx, 0), 1)

    return tokens, ContextHints(
        intensity_override=intensity_override,
        polarity_flip=polarity_flip,
        formality_boost=formality_boost,
    )


def apply_context(shade_byte: int, idx: int, hints: ContextHints) -> int:
    """Apply context modifications to a shade byte for token at ``idx``."""
    intensity = (shade_byte >> 6) & 0x03
    abstraction = (shade_byte >> 4) & 0x03
    formality = (shade_byte >> 2) & 0x03
    polarity = shade_byte & 0x03

    if idx in hints.intensity_override:
        intensity = hints.intensity_override[idx]
    if idx in hints.polarity_flip:
        # 0<->3, 1<->2 — flip across the polarity midpoint
        polarity = 3 - polarity
    if idx in hints.formality_boost:
        formality = min(3, formality + hints.formality_boost[idx])

    return (intensity << 6) | (abstraction << 4) | (formality << 2) | polarity
