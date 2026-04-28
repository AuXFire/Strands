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
_NUMBERBATCH_KIND: str = ""  # "nb19", "nb17", or ""


def _load_numberbatch():
    """Load the highest-quality Numberbatch we can find.

    Preference order:
      1. Numberbatch 19.08 English from $STRANDS_NB19_PATH or
         /tmp/numberbatch/nb19.08.txt.gz (newer, higher Spearman ρ).
      2. Numberbatch 17.06 multilingual via gensim downloader (cached).
      3. None (signals OOV-only fallback to WordNet bridge).
    """
    global _NUMBERBATCH_MODEL, _NUMBERBATCH_TRIED, _NUMBERBATCH_KIND
    if _NUMBERBATCH_TRIED:
        return _NUMBERBATCH_MODEL
    _NUMBERBATCH_TRIED = True
    if os.environ.get("STRANDS_DISABLE_NUMBERBATCH") == "1":
        return None

    # Try 19.08 first.
    nb19_path_str = os.environ.get(
        "STRANDS_NB19_PATH", "/tmp/numberbatch/nb19.08.txt.gz"
    )
    nb19_path = None
    try:
        from pathlib import Path as _Path
        p = _Path(nb19_path_str)
        if p.exists():
            nb19_path = p
    except Exception:
        nb19_path = None

    if nb19_path is not None:
        try:
            import gzip as _gzip
            import numpy as _np
            vecs: dict[str, _np.ndarray] = {}
            opener = _gzip.open if str(nb19_path).endswith(".gz") else open
            with opener(nb19_path, "rt", encoding="utf-8") as f:
                header = f.readline().split()
                dim = int(header[1])
                for line in f:
                    parts = line.rstrip().split(" ")
                    if len(parts) != dim + 1:
                        continue
                    vecs[parts[0]] = _np.asarray(parts[1:], dtype=_np.float32)
            _NUMBERBATCH_MODEL = vecs
            _NUMBERBATCH_KIND = "nb19"
            return _NUMBERBATCH_MODEL
        except Exception:
            pass

    # Fallback to 17.06.
    try:
        import gensim.downloader as api
        _NUMBERBATCH_MODEL = api.load("conceptnet-numberbatch-17-06-300")
        _NUMBERBATCH_KIND = "nb17"
    except Exception:
        _NUMBERBATCH_MODEL = None
    return _NUMBERBATCH_MODEL


def _nb_lookup(model, word: str):
    """Return the vector for ``word`` from whichever Numberbatch is loaded."""
    if model is None:
        return None
    w = word.lower().replace(" ", "_")
    if _NUMBERBATCH_KIND == "nb19":
        return model.get(w)
    # nb17 keys are /c/en/<word>
    key = f"/c/en/{w}"
    if key in model:
        return model[key]
    return None


@lru_cache(maxsize=8192)
def conceptnet_word_similarity(word_a: str, word_b: str) -> float | None:
    """Cosine similarity in ConceptNet Numberbatch.

    Returns None if the model isn't loaded or either word isn't in vocab.
    Negative cosines are clamped to 0 (treat as unrelated, not opposed).
    """
    model = _load_numberbatch()
    if model is None:
        return None

    import numpy as np

    va = _nb_lookup(model, word_a)
    vb = _nb_lookup(model, word_b)
    if va is None or vb is None:
        return None

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
        v = _nb_lookup(model, w)
        if v is None:
            continue
        vecs.append(v)
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
