"""Semantic Strands — deterministic, interpretable semantic encoding."""

from strands.code_encoder import CodeEncodeResult, detect_language, encode_code
from strands.codebook import Codebook, default_codebook
from strands.codon import DOMAIN_CODES, DOMAIN_NAMES, Codon
from strands.comparator import ComparisonResult, Match, compare_strands
from strands.encoder import EncodeResult, encode
from strands.identifier import split_identifier
from strands.index import InMemoryIndex, IndexEntry, SearchResult
from strands.shade import Shade, compute_shade, shade_similarity
from strands.strand import CodonEntry, Strand


def compare(text_a: str, text_b: str) -> ComparisonResult:
    """Convenience: encode both inputs, return alignment result."""
    return compare_strands(encode(text_a).strand, encode(text_b).strand)


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
