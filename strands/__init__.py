"""Semantic Strands — deterministic, interpretable semantic encoding.

The runtime is pure: encoding uses the codebook, comparison uses only
the strand bytes. No ConceptNet model, no Numberbatch, no GloVe, no
sidecar files. Everything needed to compare two strands lives in the
strand's 12 bytes per token.
"""

from strands.code_encoder import CodeEncodeResult, detect_language, encode_code
from strands.codebook import Codebook, default_codebook
from strands.codon import DOMAIN_CODES, DOMAIN_NAMES, Codon
from strands.comparator import ComparisonResult, Match, compare_strands
from strands.document import DocumentFingerprint, clone_similarity
from strands.encoder import EncodeResult, encode
from strands.identifier import split_identifier
from strands.index import InMemoryIndex, IndexEntry, SearchResult
from strands.phrases import add_frame_entries
from strands.relations import RelationDirection, RelationType, TypedRelation
from strands.shade import Shade, compute_shade, shade_similarity
from strands.strand import CodonEntry, Strand


def compare(
    text_a: str,
    text_b: str,
    *,
    code_aware: bool = False,
    conceptnet_bridge: bool = False,
    profile: str = "auto",
) -> ComparisonResult:
    """Encode both inputs and return their alignment score.

    Pure strand-native: no runtime models, no sidecar files, no external
    state at compare time.
    """
    return compare_strands(
        encode(text_a).strand,
        encode(text_b).strand,
        code_aware=code_aware,
        conceptnet_bridge=conceptnet_bridge,
        profile=profile,
    )


__version__ = "0.2.0"

__all__ = [
    "CodeEncodeResult",
    "Codebook",
    "Codon",
    "CodonEntry",
    "ComparisonResult",
    "DOMAIN_CODES",
    "DOMAIN_NAMES",
    "DocumentFingerprint",
    "EncodeResult",
    "InMemoryIndex",
    "IndexEntry",
    "Match",
    "SearchResult",
    "Shade",
    "Strand",
    "RelationDirection",
    "RelationType",
    "TypedRelation",
    "__version__",
    "clone_similarity",
    "add_frame_entries",
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
