"""Expand seed concepts across the full WordNet vocabulary.

Phase 1 + post-benchmark corrections C4 (better seed-sense selection) and
C5 (capped walks for top-level synsets).

Strategy:
  1. Anchor — for each seed word, prefer the synset where it is the FIRST
     lemma. Break ties by sense-frequency (lower sense number wins).
  2. Upward BFS along POS-appropriate "more general" relations with a max
     depth (default 30).
  3. Downward BFS along hyponyms / similar_tos for synsets without an
     ancestor seed. Top-level WordNet synsets (entity, abstraction,
     physical_entity, thing) are excluded as anchors so the downward pass
     doesn't pull every noun under one of them.
  4. Cross-POS derivational walk for stragglers.
  5. Fallback (AB000) so every synset contributes lemmas.

Output now carries per-lemma synset name and an `alt_codons` list of
alternative senses, supporting Correction C1 (multi-sense) and C2 (WordNet
bridge) at runtime.
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

_FALLBACK_CODON: CodonKey = ("AB", 0, 0)

# Top-level WordNet synsets that should NOT be classified directly. They
# attract nothing in the upward pass (they have no hypernyms) and would
# otherwise become anchors during the downward pass, pulling every noun
# under whatever seed first walked through them.
_TOP_LEVEL_BLOCKLIST: frozenset[str] = frozenset({
    "entity.n.01",
    "physical_entity.n.01",
    "abstraction.n.06",
    "thing.n.12",
    "object.n.01",
    "whole.n.02",
    "matter.n.03",
    "psychological_feature.n.01",
})


def _lemma_words(synset) -> Iterable[str]:
    for lemma in synset.lemmas():
        word = lemma.name().replace("_", " ").lower().strip()
        if word:
            yield word


def _select_best_synset_for_seed(word: str) -> object | None:
    """C4: prefer the synset where ``word`` is the first lemma; otherwise
    take the lowest-numbered sense (most common in WordNet's frequency order)."""
    candidates = wn.synsets(word)
    if not candidates:
        return None
    target = word.lower().replace(" ", "_")
    for synset in candidates:
        first_lemma = synset.lemmas()[0].name().lower()
        if first_lemma == target:
            return synset
    return candidates[0]


def _seed_synset_map(seeds: dict[CodonKey, list[str]]) -> dict[str, CodonKey]:
    """Map synset.name() -> codon key, preferring first-lemma synsets."""
    out: dict[str, CodonKey] = {}
    for key, words in seeds.items():
        for word in words:
            best = _select_best_synset_for_seed(word)
            if best is None:
                continue
            if best.name() not in out:
                out[best.name()] = key
            for synset in wn.synsets(word):
                if synset.name() not in out:
                    out[synset.name()] = key
    return out


def _upward_neighbors(synset) -> list:
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


def _downward_neighbors(synset, *, max_branches: int = 12) -> list:
    pos = synset.pos()
    out: list = []
    if pos in ("n", "v"):
        # C5: cap branching to avoid sucking entire subtrees through one anchor
        out += list(synset.hyponyms())[:max_branches]
        out += list(synset.instance_hyponyms())[:max_branches]
    elif pos in ("a", "s"):
        out += list(synset.also_sees())[:max_branches]
        out += list(synset.similar_tos())[:max_branches]
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


def expand_seeds_with_pos(
    seeds: dict[CodonKey, list[str]],
    *,
    max_upward_depth: int = 30,
    max_downward_depth: int = 6,
    max_senses_per_word: int = 3,
) -> tuple[
    dict[str, CodonKey],          # word -> primary codon
    dict[str, list[str]],          # word -> POS list
    dict[str, list[CodonKey]],     # word -> alt codons (C1)
    dict[str, str],                # word -> primary synset name (C2)
]:
    """Multi-sense codebook expansion. Returns four parallel maps."""
    seed_only: dict[str, CodonKey] = {}
    for key, words in seeds.items():
        for w in words:
            w_norm = w.lower().strip()
            if w_norm and w_norm not in seed_only:
                seed_only[w_norm] = key

    classified: dict[str, CodonKey] = _seed_synset_map(seeds)
    all_synsets = sorted(wn.all_synsets(), key=lambda s: s.name())

    # Pass 1: upward.
    for synset in all_synsets:
        if synset.name() in classified:
            continue
        if synset.name() in _TOP_LEVEL_BLOCKLIST:
            continue
        codon = _bfs_to_classified(
            synset, classified, neighbor_fn=_upward_neighbors, max_depth=max_upward_depth
        )
        if codon is not None:
            classified[synset.name()] = codon

    # Pass 2: downward — but skip top-level synsets so they don't anchor
    # huge subtrees.
    for synset in all_synsets:
        if synset.name() in classified:
            continue
        if synset.name() in _TOP_LEVEL_BLOCKLIST:
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
        for lemma in synset.lemmas():
            for d in lemma.derivationally_related_forms():
                if d.synset().name() in classified:
                    classified[synset.name()] = classified[d.synset().name()]
                    break
            if synset.name() in classified:
                break

    # Pass 4: fallback for everything else (including top-level entries).
    for synset in all_synsets:
        if synset.name() not in classified:
            classified[synset.name()] = _FALLBACK_CODON

    word_to_codon: dict[str, CodonKey] = dict(seed_only)
    word_to_pos: dict[str, list[str]] = {}
    word_to_alts: dict[str, list[CodonKey]] = {}
    word_to_synset: dict[str, str] = {}

    # Pre-populate synsets for seed words via WordNet so the C2 bridge
    # has something to look up at runtime.
    for word in seed_only:
        best = _select_best_synset_for_seed(word)
        if best is not None:
            word_to_synset[word] = best.name()

    for synset in all_synsets:
        codon = classified[synset.name()]
        pos = synset.pos()
        for lemma in _lemma_words(synset):
            if lemma not in word_to_codon:
                word_to_codon[lemma] = codon
                word_to_synset[lemma] = synset.name()
            else:
                # Track this as an alternative sense (C1).
                primary = word_to_codon[lemma]
                if codon != primary:
                    alts = word_to_alts.setdefault(lemma, [])
                    if codon not in alts and len(alts) < max_senses_per_word - 1:
                        alts.append(codon)
                # Pick up a synset name if this word didn't have one (seeds).
                if lemma not in word_to_synset:
                    word_to_synset[lemma] = synset.name()
            pos_list = word_to_pos.setdefault(lemma, [])
            if pos not in pos_list:
                pos_list.append(pos)

    return word_to_codon, word_to_pos, word_to_alts, word_to_synset


def expand_seeds(
    seeds: dict[CodonKey, list[str]],
    **kwargs,
) -> dict[str, CodonKey]:
    """Backwards-compatible wrapper that drops the alt/POS/synset metadata."""
    primary, _, _, _ = expand_seeds_with_pos(seeds, **kwargs)
    return primary
