"""Runtime relatedness signals (Tier 2 + Tier 3 of the comparator).

Tier 2 — WordNet bridge:
  Wu-Palmer similarity over the cross-product of synsets for two words.
  Pure WordNet data (already a dependency for codebook construction).

Tier 3 — ConceptNet bridge (optional):
  ConceptNet Numberbatch (gensim model) cosine between word vectors.
  Loaded on first use; lookup is keyed by ``/c/en/<word>``.

Both tiers are gated by explicit flags on compare_strands() — the spec's
4-byte primary representation (codon+shade) is unchanged. These signals
augment cross-domain comparison only when the codon-level score is low
and the relevant database is available.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

# ---- Tier 2: WordNet --------------------------------------------------------


@lru_cache(maxsize=8192)
def wordnet_similarity(syn_a: str, syn_b: str) -> float | None:
    """Cross-product Wu-Palmer over candidate synsets for two synset names.

    Returns the maximum similarity across the cross product, or None on any
    failure (missing synsets, library errors).
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None

    try:
        s_a = wn.synset(syn_a)
        s_b = wn.synset(syn_b)
    except Exception:
        return None

    best = 0.0
    # First try same-POS comparison (Wu-Palmer requires same POS).
    if s_a.pos() == s_b.pos():
        try:
            sim = s_a.wup_similarity(s_b)
            if sim is not None and sim > best:
                best = float(sim)
        except Exception:
            pass

    # Path similarity also handles cross-POS via the unified taxonomy.
    try:
        sim = s_a.path_similarity(s_b)
        if sim is not None and sim > best:
            best = float(sim)
    except Exception:
        pass

    return best if best > 0 else None


@lru_cache(maxsize=4096)
def wordnet_word_similarity(word_a: str, word_b: str) -> float | None:
    """Best similarity across all synsets of two words. Slower than the
    synset-keyed variant but works without precomputed synset names."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None

    syns_a = wn.synsets(word_a)
    syns_b = wn.synsets(word_b)
    if not syns_a or not syns_b:
        return None

    best = 0.0
    for s_a in syns_a[:5]:
        for s_b in syns_b[:5]:
            try:
                if s_a.pos() == s_b.pos():
                    sim = s_a.wup_similarity(s_b)
                    if sim is not None and sim > best:
                        best = float(sim)
                sim = s_a.path_similarity(s_b)
                if sim is not None and sim > best:
                    best = float(sim)
            except Exception:
                continue
    return best if best > 0 else None


# ---- Tier 3: ConceptNet / Numberbatch ---------------------------------------

_NUMBERBATCH_MODEL: Any = None
_NUMBERBATCH_TRIED: bool = False


def _load_numberbatch():
    global _NUMBERBATCH_MODEL, _NUMBERBATCH_TRIED
    if _NUMBERBATCH_TRIED:
        return _NUMBERBATCH_MODEL
    _NUMBERBATCH_TRIED = True
    if os.environ.get("STRANDS_DISABLE_NUMBERBATCH") == "1":
        return None
    try:
        import gensim.downloader as api
        _NUMBERBATCH_MODEL = api.load("conceptnet-numberbatch-17-06-300")
    except Exception:
        _NUMBERBATCH_MODEL = None
    return _NUMBERBATCH_MODEL


@lru_cache(maxsize=8192)
def conceptnet_word_similarity(word_a: str, word_b: str) -> float | None:
    """Cosine similarity in ConceptNet Numberbatch.

    Returns None if the model isn't loaded or either word isn't in vocab.
    Negative cosines are clamped to 0 (treat as unrelated, not opposed).
    """
    model = _load_numberbatch()
    if model is None:
        return None

    key_a = f"/c/en/{word_a.lower().replace(' ', '_')}"
    key_b = f"/c/en/{word_b.lower().replace(' ', '_')}"
    if key_a not in model or key_b not in model:
        return None

    import numpy as np

    va = model[key_a]
    vb = model[key_b]
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return None
    cos = float(np.dot(va, vb) / denom)
    return max(0.0, cos)


def is_conceptnet_available() -> bool:
    return _load_numberbatch() is not None


def conceptnet_mean_vector(words: list[str], *, sif: bool = True, sif_a: float = 1e-3):
    """Mean ConceptNet vector across the words.

    When ``sif=True`` (default), apply smooth-inverse-frequency weighting
    (Arora et al. 2017): weight = a / (a + p(w)) where p(w) is unigram
    probability. Down-weights frequent function words for sentence-level
    similarity. Significantly outperforms naive mean on STS.
    """
    model = _load_numberbatch()
    if model is None:
        return None
    import numpy as np

    if sif:
        try:
            from wordfreq import word_frequency
        except ImportError:
            sif = False

    vecs = []
    weights = []
    for w in words:
        if not w:
            continue
        key = f"/c/en/{w.lower().replace(' ', '_')}"
        if key not in model:
            continue
        vecs.append(model[key])
        if sif:
            p = max(1e-7, word_frequency(w, "en"))
            weights.append(sif_a / (sif_a + p))
        else:
            weights.append(1.0)
    if not vecs:
        return None
    arr = np.array(vecs)
    w_arr = np.array(weights)
    return (arr * w_arr[:, None]).sum(axis=0) / w_arr.sum()


def vector_cosine(va, vb) -> float:
    import numpy as np
    if va is None or vb is None:
        return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    if d == 0:
        return 0.0
    return float(np.dot(va, vb) / d)
