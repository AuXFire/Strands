"""Filter words by Zipf frequency to keep the codebook focused on common English."""

from __future__ import annotations

try:
    from wordfreq import zipf_frequency
except ImportError as e:
    raise ImportError(
        "wordfreq is required. Install with `pip install wordfreq`."
    ) from e


def is_common(word: str, threshold: float = 2.0) -> bool:
    """Return True if word's Zipf frequency >= threshold.

    Zipf scale: 1 = very rare, 4 = common, 7 = ultra-frequent ("the").
    Default 2.0 keeps roughly the top 30-50k English words.
    """
    return zipf_frequency(word, "en") >= threshold


def formality_from_frequency(word: str) -> int:
    """Heuristic: rarer words tend to be more formal/academic."""
    z = zipf_frequency(word, "en")
    if z >= 5.0:
        return 0  # casual
    if z >= 4.0:
        return 1  # neutral
    if z >= 3.0:
        return 2  # formal
    return 3  # academic/rare
