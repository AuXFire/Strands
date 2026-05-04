"""BDRM Semantic Backbone — the compiled, fixed-width binary knowledge graph.

The backbone is the world-knowledge layer of BDRM. It holds:
  - 128-byte concept nodes drawn from WordNet synsets (and ConceptNet
    concepts in later milestones).
  - 32-byte typed weighted edges drawn from WordNet relations
    (hypernym, hyponym, meronym, antonym, similar) and ConceptNet's
    /r/* relation taxonomy.
  - A variable-width lemma table mapping each node to its surface forms.

It is built once (via ``scripts/build_backbone.py``) and memory-mapped
at runtime.
"""

from strands.backbone.compute_module import (
    AnchorFact,
    ComputeModule,
    Conditioning,
    Fact,
    HistoryTurn,
    StubComputeModule,
    build_conditioning,
)
from strands.backbone.inference import (
    InferenceConfig,
    InferenceResult,
    TokenCandidate,
    classify_intent,
    disambiguate,
    extract_subgraph,
    infer,
    map_tokens_to_candidates,
    spread_activation,
    tag_uncertain,
)
from strands.backbone.loader import Backbone, BackboneNode, load
from strands.backbone.response import (
    DiscourseState,
    Response,
    TurnRecord,
    respond,
)
from strands.backbone.schema import (
    EDGE_DTYPE,
    NODE_DTYPE,
    ConceptType,
    Rel,
    Source,
)

__all__ = [
    "AnchorFact",
    "Backbone",
    "BackboneNode",
    "ComputeModule",
    "ConceptType",
    "Conditioning",
    "DiscourseState",
    "Fact",
    "HistoryTurn",
    "TurnRecord",
    "EDGE_DTYPE",
    "InferenceConfig",
    "InferenceResult",
    "NODE_DTYPE",
    "Rel",
    "Response",
    "Source",
    "StubComputeModule",
    "TokenCandidate",
    "build_conditioning",
    "classify_intent",
    "disambiguate",
    "extract_subgraph",
    "infer",
    "load",
    "map_tokens_to_candidates",
    "respond",
    "spread_activation",
    "tag_uncertain",
]
