"""Expand seed concepts into a full word→codon map via WordNet.

For each seed word:
- Find its WordNet synsets.
- Add every lemma in the synset → seed codon.
- Walk direct hyponyms (1 level) and their lemmas → seed codon.
- Walk direct synonyms via shared synsets.

The first seed word in a concept's list is treated as the canonical anchor for
synset selection (we take its first noun/verb/adj synset depending on what's
available).
"""

from __future__ import annotations

from typing import Iterable

try:
    from nltk.corpus import wordnet as wn
except ImportError as e:
    raise ImportError(
        "nltk is required for codebook construction. Install with `pip install nltk`."
    ) from e


def _lemma_words(synset) -> Iterable[str]:
    for lemma in synset.lemmas():
        word = lemma.name().replace("_", " ").lower()
        if word and "_" not in word:
            yield word


def _primary_synset(word: str):
    candidates = wn.synsets(word)
    if not candidates:
        return None
    return candidates[0]


def expand_seeds(
    seeds: dict[tuple[str, int, int], list[str]],
    hyponym_depth: int = 1,
) -> dict[str, tuple[str, int, int]]:
    """Return word → (domain_code, category, concept) mapping.

    On collision, the first-assigned mapping wins (seed primacy).
    """
    word_to_codon: dict[str, tuple[str, int, int]] = {}

    # First pass: seed words themselves (highest priority, never overridden)
    for key, words in seeds.items():
        for w in words:
            w_norm = w.lower().strip()
            if w_norm and w_norm not in word_to_codon:
                word_to_codon[w_norm] = key

    # Second pass: WordNet expansion
    for key, words in seeds.items():
        for word in words:
            synset = _primary_synset(word)
            if synset is None:
                continue
            for lemma in _lemma_words(synset):
                if lemma not in word_to_codon:
                    word_to_codon[lemma] = key

            # Hyponyms (more specific words inherit the seed's codon)
            frontier = [synset]
            for _ in range(hyponym_depth):
                next_frontier = []
                for s in frontier:
                    for hypo in s.hyponyms():
                        for lemma in _lemma_words(hypo):
                            if lemma not in word_to_codon:
                                word_to_codon[lemma] = key
                        next_frontier.append(hypo)
                frontier = next_frontier

    return word_to_codon
