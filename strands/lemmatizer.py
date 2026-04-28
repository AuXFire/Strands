"""WordNet-backed lemmatization with POS inference."""

from __future__ import annotations

from functools import lru_cache


def _ensure_nltk():
    try:
        from nltk.corpus import wordnet  # noqa: F401
        from nltk.stem import WordNetLemmatizer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nltk is required. Install with `pip install nltk`."
        ) from e


_ensure_nltk()

from nltk import pos_tag  # noqa: E402
from nltk.corpus import wordnet  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

_LEMMATIZER = WordNetLemmatizer()


def _wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


@lru_cache(maxsize=10000)
def lemmatize(word: str) -> str:
    word = word.lower()
    try:
        tagged = pos_tag([word])
        pos = _wordnet_pos(tagged[0][1])
    except Exception:
        pos = wordnet.NOUN
    return _LEMMATIZER.lemmatize(word, pos)


def lemmatize_all(words: list[str]) -> list[str]:
    return [lemmatize(w) for w in words]
