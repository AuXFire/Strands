"""Surface realization layer (BDRM §3.5).

Phrasal templates with named slots, where each slot can be filled
deterministically (from the backbone) or deferred to the compute
module. The slot/compute interchange happens here: templates produce
a stream of Segments, each tagged with its source so the caller can
emit deterministic segments immediately and pause for compute on the
rest.
"""

from strands.realization.render import (
    FillerValue,
    render,
    render_text,
)
from strands.realization.seeds import (
    SEED_TEMPLATES,
    build_default_store,
)
from strands.realization.store import TemplateStore
from strands.realization.template import (
    Register,
    Rendered,
    Segment,
    SlotKind,
    SlotSpec,
    Template,
)

__all__ = [
    "FillerValue",
    "Register",
    "Rendered",
    "SEED_TEMPLATES",
    "Segment",
    "SlotKind",
    "SlotSpec",
    "Template",
    "TemplateStore",
    "build_default_store",
    "render",
    "render_text",
]
