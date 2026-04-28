"""Expand seed concepts across the full WordNet vocabulary.

Strategy (all passes operate on synset.name()):
  1. Anchor: every seed word's synsets are mapped to its codon.
  2. Upward pass — for every unclassified synset, BFS along POS-appropriate
     "more general" relations (hypernyms / similar_tos / pertainyms /
     derivationally_related_forms). Stop at the first classified synset.
  3. Downward pass — for synsets still unclassified after pass 2, BFS along
     the inverse relations (hyponyms / also_sees / similar_tos in reverse)
     until hitting a classified descendant.
  4. Lateral derivational pass — follow derivationally_related_forms across
     POS boundaries (e.g. an adverb's adjective root, an adjective's noun
     pertainym).
  5. Fallback — anything still unclassified gets the AB (Abstract) generic
     codon so every synset contributes lemmas to the codebook.

Every classified synset emits one codebook entry per lemma, with multi-word
phrases joined by spaces. First-classified-wins on lemma collisions.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

try:
    from nltk.corpus import wordnet as wn
except ImportError as e:
    raise ImportError(
        "nltk is required for codebook construction. Install with `pip install nltk`."
    ) from e

CodonKey = tuple[str, int, int]

# Default fallback codon for synsets with no path to any seed.
_FALLBACK_CODON: CodonKey = ("AB", 0, 0)


def _lemma_words(synset) -> Iterable[str]:
    for lemma in synset.lemmas():
        word = lemma.name().replace("_", " ").lower().strip()
        if word:
            yield word


def _seed_synset_map(seeds: dict[CodonKey, list[str]]) -> dict[str, CodonKey]:
    out: dict[str, CodonKey] = {}
    for key, words in seeds.items():
        for word in words:
            for synset in wn.synsets(word):
                if synset.name() not in out:
                    out[synset.name()] = key
    return out


def _upward_neighbors(synset) -> list:
    """POS-appropriate "more general" neighbors."""
    pos = synset.pos()
    out: list = []
    if pos in ("n", "v"):
        out += list(synset.hypernyms())
        out += list(synset.instance_hypernyms())
        if pos == "v":
            out += list(synset.verb_groups())
    elif pos in ("a", "s"):
        out += list(synset.similar_tos())
        for lemma in synset.lemmas():
            out += [a.synset() for a in lemma.antonyms()]
            out += [p.synset() for p in lemma.pertainyms()]
            out += [d.synset() for d in lemma.derivationally_related_forms()]
    elif pos == "r":
        for lemma in synset.lemmas():
            out += [p.synset() for p in lemma.pertainyms()]
            out += [d.synset() for d in lemma.derivationally_related_forms()]
    return out


def _downward_neighbors(synset) -> list:
    pos = synset.pos()
    out: list = []
    if pos in ("n", "v"):
        out += list(synset.hyponyms())
        out += list(synset.instance_hyponyms())
    elif pos in ("a", "s"):
        # also_sees and similar_tos are bidirectional in spirit
        out += list(synset.also_sees())
        out += list(synset.similar_tos())
    elif pos == "r":
        out += list(synset.also_sees()) if hasattr(synset, "also_sees") else []
    # Always look at derivational forms — they bridge POS for everyone.
    for lemma in synset.lemmas():
        out += [d.synset() for d in lemma.derivationally_related_forms()]
    return out


def _bfs_to_classified(
    synset,
    classified: dict[str, CodonKey],
    *,
    neighbor_fn,
    max_depth: int,
) -> CodonKey | None:
    if synset.name() in classified:
        return classified[synset.name()]

    visited: set[str] = {synset.name()}
    frontier: deque = deque([(synset, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for nb in neighbor_fn(node):
            if nb is None or nb.name() in visited:
                continue
            visited.add(nb.name())
            if nb.name() in classified:
                return classified[nb.name()]
            frontier.append((nb, depth + 1))
    return None


def expand_seeds(
    seeds: dict[CodonKey, list[str]],
    *,
    max_upward_depth: int = 30,
    max_downward_depth: int = 12,
    use_fallback: bool = True,
) -> dict[str, CodonKey]:
    """Return word -> codon for the entire WordNet vocabulary."""
    word_to_codon: dict[str, CodonKey] = {}

    # Pass 0: explicit seed words always win.
    for key, words in seeds.items():
        for w in words:
            w_norm = w.lower().strip()
            if w_norm and w_norm not in word_to_codon:
                word_to_codon[w_norm] = key

    classified: dict[str, CodonKey] = _seed_synset_map(seeds)

    all_synsets = sorted(wn.all_synsets(), key=lambda s: s.name())

    # Pass 1: walk UP from each unclassified synset.
    for synset in all_synsets:
        if synset.name() in classified:
            continue
        codon = _bfs_to_classified(
            synset, classified, neighbor_fn=_upward_neighbors, max_depth=max_upward_depth
        )
        if codon is not None:
            classified[synset.name()] = codon

    # Pass 2: walk DOWN from each still-unclassified synset.
    for synset in all_synsets:
        if synset.name() in classified:
            continue
        codon = _bfs_to_classified(
            synset, classified, neighbor_fn=_downward_neighbors, max_depth=max_downward_depth
        )
        if codon is not None:
            classified[synset.name()] = codon

    # Pass 3: cross-POS derivational walk.
    for synset in all_synsets:
        if synset.name() in classified:
            continue
        # Try following derivationally_related_forms across POS aggressively.
        related: list = []
        for lemma in synset.lemmas():
            related += [d.synset() for d in lemma.derivationally_related_forms()]
        for r in related:
            if r.name() in classified:
                classified[synset.name()] = classified[r.name()]
                break

    # Pass 4: fallback for anything that survived all passes.
    if use_fallback:
        for synset in all_synsets:
            if synset.name() not in classified:
                classified[synset.name()] = _FALLBACK_CODON

    # Emit lemma -> codon for every classified synset, tracking POS so the
    # caller can derive inflectional variants per POS.
    for synset in all_synsets:
        codon = classified.get(synset.name())
        if codon is None:
            continue
        for lemma in _lemma_words(synset):
            if lemma not in word_to_codon:
                word_to_codon[lemma] = codon

    return word_to_codon


def expand_seeds_with_pos(
    seeds: dict[CodonKey, list[str]],
    *,
    max_upward_depth: int = 30,
    max_downward_depth: int = 12,
) -> tuple[dict[str, CodonKey], dict[str, list[str]]]:
    """Like ``expand_seeds`` but also returns word -> [POS codes] for inflection."""
    seed_only: dict[str, CodonKey] = {}
    for key, words in seeds.items():
        for w in words:
            w_norm = w.lower().strip()
            if w_norm and w_norm not in seed_only:
                seed_only[w_norm] = key

    classified: dict[str, CodonKey] = _seed_synset_map(seeds)
    all_synsets = sorted(wn.all_synsets(), key=lambda s: s.name())

    for synset in all_synsets:
        if synset.name() in classified:
            continue
        codon = _bfs_to_classified(
            synset, classified, neighbor_fn=_upward_neighbors, max_depth=max_upward_depth
        )
        if codon is not None:
            classified[synset.name()] = codon

    for synset in all_synsets:
        if synset.name() in classified:
            continue
        codon = _bfs_to_classified(
            synset, classified, neighbor_fn=_downward_neighbors, max_depth=max_downward_depth
        )
        if codon is not None:
            classified[synset.name()] = codon

    for synset in all_synsets:
        if synset.name() in classified:
            continue
        for lemma in synset.lemmas():
            for d in lemma.derivationally_related_forms():
                if d.synset().name() in classified:
                    classified[synset.name()] = classified[d.synset().name()]
                    break
            if synset.name() in classified:
                break

    for synset in all_synsets:
        if synset.name() not in classified:
            classified[synset.name()] = _FALLBACK_CODON

    word_to_codon: dict[str, CodonKey] = dict(seed_only)
    word_to_pos: dict[str, list[str]] = {}

    for synset in all_synsets:
        codon = classified[synset.name()]
        pos = synset.pos()
        for lemma in _lemma_words(synset):
            if lemma not in word_to_codon:
                word_to_codon[lemma] = codon
            pos_list = word_to_pos.setdefault(lemma, [])
            if pos not in pos_list:
                pos_list.append(pos)

    return word_to_codon, word_to_pos
