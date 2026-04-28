"""Generate English inflectional variants for codebook entries.

Each base lemma gets simple rule-based inflections based on its POS, plus
WordNet's exception lists for irregular forms. Variants inherit the base
lemma's codon. First-classified-wins on collisions.
"""

from __future__ import annotations

from collections.abc import Iterable

try:
    from nltk.corpus import wordnet as wn
except ImportError as e:
    raise ImportError(
        "nltk is required for morphology. Install with `pip install nltk`."
    ) from e

VOWELS = set("aeiou")


def _ends_with_any(word: str, suffixes: Iterable[str]) -> bool:
    return any(word.endswith(s) for s in suffixes)


def noun_plural(word: str) -> str | None:
    if " " in word or len(word) < 2:
        return None
    if _ends_with_any(word, ("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) >= 2 and word[-2] not in VOWELS:
        return word[:-1] + "ies"
    if word.endswith("f") and len(word) >= 3:
        return word[:-1] + "ves"
    if word.endswith("fe"):
        return word[:-2] + "ves"
    return word + "s"


def verb_third_person(word: str) -> str | None:
    return noun_plural(word)


def verb_past(word: str) -> str | None:
    if " " in word or len(word) < 2:
        return None
    if word.endswith("e"):
        return word + "d"
    if word.endswith("y") and len(word) >= 2 and word[-2] not in VOWELS:
        return word[:-1] + "ied"
    # double final consonant for short CVC verbs
    if (
        len(word) >= 3
        and word[-1] not in VOWELS
        and word[-1] not in "wxy"
        and word[-2] in VOWELS
        and word[-3] not in VOWELS
    ):
        return word + word[-1] + "ed"
    return word + "ed"


def verb_gerund(word: str) -> str | None:
    if " " in word or len(word) < 2:
        return None
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith("ee"):
        return word[:-1] + "ing"
    if (
        len(word) >= 3
        and word[-1] not in VOWELS
        and word[-1] not in "wxy"
        and word[-2] in VOWELS
        and word[-3] not in VOWELS
    ):
        return word + word[-1] + "ing"
    return word + "ing"


def adjective_comparative(word: str) -> str | None:
    if " " in word or len(word) < 3 or len(word) > 8:
        return None
    if word.endswith("y") and word[-2] not in VOWELS:
        return word[:-1] + "ier"
    if word.endswith("e"):
        return word + "r"
    if (
        len(word) >= 3
        and word[-1] not in VOWELS
        and word[-1] not in "wxy"
        and word[-2] in VOWELS
        and word[-3] not in VOWELS
    ):
        return word + word[-1] + "er"
    return word + "er"


def adjective_superlative(word: str) -> str | None:
    if " " in word or len(word) < 3 or len(word) > 8:
        return None
    if word.endswith("y") and word[-2] not in VOWELS:
        return word[:-1] + "iest"
    if word.endswith("e"):
        return word + "st"
    if (
        len(word) >= 3
        and word[-1] not in VOWELS
        and word[-1] not in "wxy"
        and word[-2] in VOWELS
        and word[-3] not in VOWELS
    ):
        return word + word[-1] + "est"
    return word + "est"


def adverb_from_adjective(word: str) -> str | None:
    if " " in word or len(word) < 3:
        return None
    if word.endswith("ly"):
        return None
    if word.endswith("y") and word[-2] not in VOWELS:
        return word[:-1] + "ily"
    if word.endswith("ic"):
        return word + "ally"
    if word.endswith("le") and len(word) >= 4 and word[-3] not in VOWELS:
        return word[:-1] + "y"
    return word + "ly"


def _looks_like_inflection(word: str) -> bool:
    """Heuristic: skip generating inflections from already-inflected forms.

    Suffixes like ``-ing``, ``-ed``, ``-est``, ``-ly``, ``-ies`` are usually
    inflections. We deliberately do NOT include ``-er`` because many base
    nouns end in -er (computer, server, hammer, writer).
    """
    if " " in word:
        return False
    return word.endswith(("ing", "ed", "est", "ly", "ies"))


def variants_for(word: str, pos: str) -> list[str]:
    """Return rule-based inflectional variants for a word and POS code (n/v/a/r)."""
    if _looks_like_inflection(word):
        return []
    variants: list[str] = []
    if pos == "n":
        v = noun_plural(word)
        if v:
            variants.append(v)
    elif pos == "v":
        for fn in (verb_third_person, verb_past, verb_gerund):
            v = fn(word)
            if v:
                variants.append(v)
    elif pos in ("a", "s"):
        for fn in (adjective_comparative, adjective_superlative, adverb_from_adjective):
            v = fn(word)
            if v:
                variants.append(v)
    return [v for v in variants if v != word]


def wordnet_irregular_forms() -> dict[str, list[str]]:
    """Return inflected_form -> [lemma] from WordNet's exception lists."""
    out: dict[str, list[str]] = {}
    for pos in ("n", "v", "a", "r"):
        try:
            exc = wn._exception_map.get(pos, {})  # type: ignore[attr-defined]
        except AttributeError:
            exc = {}
        for inflected, lemmas in exc.items():
            inflected_norm = inflected.replace("_", " ").lower()
            out.setdefault(inflected_norm, []).extend(
                lemma.replace("_", " ").lower() for lemma in lemmas
            )
    return out
