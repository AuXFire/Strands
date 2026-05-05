"""Surface realization layer (BDRM §3.5).

Phrasal templates with named slots, where each slot can be filled
deterministically (from the backbone) or deferred to the compute
module. The slot/compute interchange happens here: templates produce
a stream of Segments, each tagged with its source so the caller can
emit deterministic segments immediately and pause for compute on the
rest.

Tree-level structure (BDRM §3.4.3): planners return a
``ResponseStructure`` — an ordered tree of ``CommunicativeAct``s with
``ConceptualLeaf``s at the leaves. ``realize(structure, store)`` walks
the tree depth-first and applies per-leaf surface realization.
"""

from strands.realization.realize import RealizedResponse, realize
from strands.realization.render import (
    FillerValue,
    render,
    render_text,
)
from strands.realization.seeds import (
    build_default_store,
    seed_templates,
)
from strands.realization.store import TemplateStore
from strands.realization.structure import (
    ActType,
    CommunicativeAct,
    ConceptualLeaf,
    ResponseStructure,
    sequence,
    single_leaf,
)
from strands.realization.template import (
    Register,
    Rendered,
    Segment,
    SlotKind,
    SlotSpec,
    Template,
)

__all__ = [
    "ActType",
    "CommunicativeAct",
    "ConceptualLeaf",
    "FillerValue",
    "RealizedResponse",
    "Register",
    "Rendered",
    "ResponseStructure",
    "Segment",
    "SlotKind",
    "SlotSpec",
    "Template",
    "TemplateStore",
    "build_default_store",
    "realize",
    "render",
    "render_text",
    "seed_templates",
    "sequence",
    "single_leaf",
]

