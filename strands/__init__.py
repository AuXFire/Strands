"""Semantic Strands — deterministic, interpretable semantic encoding."""

from strands.code_encoder import CodeEncodeResult, detect_language, encode_code
from strands.codebook import Codebook, default_codebook
from strands.codon import DOMAIN_CODES, DOMAIN_NAMES, Codon
from strands.comparator import ComparisonResult, Match, compare_strands
from strands.document import DocumentFingerprint, clone_similarity
from strands.encoder import EncodeResult, encode
from strands.identifier import split_identifier
from strands.index import InMemoryIndex, IndexEntry, SearchResult
from strands.shade import Shade, compute_shade, shade_similarity
from strands.strand import CodonEntry, Strand


def compare(
    text_a: str,
    text_b: str,
    *,
    wordnet_bridge: bool = True,
    conceptnet_bridge: bool | None = None,
    sentence_mode: bool | None = None,
) -> ComparisonResult:
    """Convenience: encode both inputs, return alignment result.

    For sentence-length inputs (auto-detected when ConceptNet is enabled
    and either input has more than ~4 content tokens), uses ConceptNet
    mean-vector cosine over the raw tokens — the same baseline that beats
    GloVe-mean on STS, while strand storage remains 4 bytes/word.
    """
    import os
    cn_default = os.environ.get("STRANDS_CONCEPTNET", "0") == "1"
    cn_enabled = conceptnet_bridge if conceptnet_bridge is not None else cn_default

    auto_sentence = (
        cn_enabled
        and sentence_mode is not False
        and (len(text_a.split()) > 4 or len(text_b.split()) > 4)
    )

    if (sentence_mode or auto_sentence) and cn_enabled:
        import re
        from strands.relatedness import (
            conceptnet_mean_vector,
            is_conceptnet_available,
            vector_cosine,
        )

        if is_conceptnet_available():
            tok_re = re.compile(r"[A-Za-z']+")
            words_a = [t.lower() for t in tok_re.findall(text_a)]
            words_b = [t.lower() for t in tok_re.findall(text_b)]
            va = conceptnet_mean_vector(words_a)
            vb = conceptnet_mean_vector(words_b)
            if va is not None and vb is not None:
                score = max(0.0, vector_cosine(va, vb))
                return ComparisonResult(score=score, matches=[],
                                        unmatched_a=[], unmatched_b=[])

    return compare_strands(
        encode(text_a).strand,
        encode(text_b).strand,
        wordnet_bridge=wordnet_bridge,
        conceptnet_bridge=conceptnet_bridge,
        sentence_mode=sentence_mode,
    )


__version__ = "0.1.0"

__all__ = [
    "CodeEncodeResult",
    "Codebook",
    "Codon",
    "CodonEntry",
    "ComparisonResult",
    "DOMAIN_CODES",
    "DOMAIN_NAMES",
    "EncodeResult",
    "InMemoryIndex",
    "IndexEntry",
    "Match",
    "SearchResult",
    "Shade",
    "Strand",
    "__version__",
    "compare",
    "compare_strands",
    "compute_shade",
    "default_codebook",
    "detect_language",
    "encode",
    "encode_code",
    "shade_similarity",
    "split_identifier",
]
