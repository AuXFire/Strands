"""Compute polarity hint per word using SentiWordNet scores.

Returns a polarity bit (0-3) where:
  0 = clearly negative
  1 = slightly negative / neutral-low
  2 = neutral / slightly positive
  3 = clearly positive
"""

from __future__ import annotations

try:
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn
except ImportError as e:
    raise ImportError(
        "nltk is required. Install with `pip install nltk`."
    ) from e


def _score_word(word: str) -> tuple[float, float] | None:
    synsets = wn.synsets(word)
    if not synsets:
        return None

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for s in synsets[:3]:  # average top 3 senses
        try:
            sw = swn.senti_synset(s.name())
        except Exception:
            continue
        pos_scores.append(sw.pos_score())
        neg_scores.append(sw.neg_score())
    if not pos_scores:
        return None
    return sum(pos_scores) / len(pos_scores), sum(neg_scores) / len(neg_scores)


def polarity_bits(word: str) -> int:
    scores = _score_word(word)
    if scores is None:
        return 1  # default neutral-low
    pos, neg = scores
    net = pos - neg
    if net <= -0.25:
        return 0
    if net < 0:
        return 1
    if net < 0.25:
        return 2
    return 3
