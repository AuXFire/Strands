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

from strands.backbone.loader import Backbone, BackboneNode, load
from strands.backbone.schema import (
    EDGE_DTYPE,
    NODE_DTYPE,
    ConceptType,
    Rel,
    Source,
)

__all__ = [
    "Backbone",
    "BackboneNode",
    "ConceptType",
    "EDGE_DTYPE",
    "NODE_DTYPE",
    "Rel",
    "Source",
    "load",
]
