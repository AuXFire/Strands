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
    """Cosine similarity from ConceptNet vectors.

    Strategy:
      1. Try the codebook's embedded per-word vectors (PCA-reduced,
         int8-quantized, ~16 MB sidecar). Pure strand-native, no runtime
         model dependency.
      2. If embedded vectors aren't present for one or both words, fall
         back to the runtime Numberbatch model (only loaded if available).
      3. Return None if neither path can resolve both words.
    """
    import numpy as np

    # Tier 1: codebook-embedded vectors. Preferred — these ship with the
    # codebook, no runtime model load needed.
    try:
        from strands.codebook import default_codebook
        cb = default_codebook()
    except Exception:
        cb = None

    if cb is not None and cb.has_embedded_vectors:
        va = cb.embedded_vector(word_a)
        vb = cb.embedded_vector(word_b)
        if va is not None and vb is not None:
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            if denom > 0:
                cos = float(np.dot(va, vb) / denom)
                return max(0.0, cos)

    # Tier 2: runtime Numberbatch (only if available + opted in).
    model = _load_numberbatch()
    if model is None:
        return None

    rva = _nb_lookup(model, word_a)
    rvb = _nb_lookup(model, word_b)
    if rva is None or rvb is None:
        return None

    denom = np.linalg.norm(rva) * np.linalg.norm(rvb)
    if denom == 0:
        return None
    cos = float(np.dot(rva, rvb) / denom)
    return max(0.0, cos)


def is_conceptnet_available() -> bool:
    """True if either the codebook ships embedded vectors *or* the runtime
    model is loadable. Embedded vectors take precedence."""
    try:
        from strands.codebook import default_codebook
        if default_codebook().has_embedded_vectors:
            return True
    except Exception:
        pass
    return _load_numberbatch() is not None


def conceptnet_mean_vector(words: list[str], *, sif: bool = True, sif_a: float = 1e-3):
    """Mean ConceptNet vector across the words.

    Prefers codebook-embedded vectors (no runtime model). Falls back to
    runtime Numberbatch when a word is missing from the embedding table.

    When ``sif=True`` (default), apply smooth-inverse-frequency weighting
    (Arora et al. 2017): weight = a / (a + p(w)) where p(w) is unigram
    probability.
    """
    import numpy as np

    if sif:
        try:
            from wordfreq import word_frequency
        except ImportError:
            sif = False

    # Try codebook-embedded vectors first.
    cb = None
    try:
        from strands.codebook import default_codebook
        cb = default_codebook()
        if not cb.has_embedded_vectors:
            cb = None
    except Exception:
        cb = None

    model = _load_numberbatch() if cb is None else None  # only load if no CB embeddings

    vecs = []
    weights = []
    for w in words:
        if not w:
            continue
        v = None
        if cb is not None:
            v = cb.embedded_vector(w)
        if v is None and model is not None:
            v = _nb_lookup(model, w)
        if v is None and model is None:
            # Last resort: try loading runtime model now.
            model = _load_numberbatch()
            if model is not None:
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
    # Vectors may be different dimensions if mixed (embedded 64-d vs
    # runtime 300-d). Stick to the first dimensionality seen.
    target_dim = len(vecs[0])
    vecs = [v for v in vecs if len(v) == target_dim]
    weights = weights[: len(vecs)]
    arr = np.asarray(vecs)
    w_arr = np.asarray(weights)
    return (arr * w_arr[:, None]).sum(axis=0) / w_arr.sum()


def vector_cosine(va, vb) -> float:
    import numpy as np
    if va is None or vb is None:
        return 0.0
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    if d == 0:
        return 0.0
    return float(np.dot(va, vb) / d)
