"""Response structure tree (BDRM §3.4.3).

The deterministic answerer used to return a flat string. Per spec
the response formulation step assembles selected content into a
**response structure** — an ordered tree of communicative acts with
conceptual content at the leaves. Surface realization (§3.5) walks
the tree per leaf, picking template / lemma / composed / compute for
each one.

This module defines the tree types. Surface realization that walks
them is in ``strands/realization/realize.py``.

Ordering principles the planner is expected to respect when building
trees (informally for now; learned later in §5.1):
  - given-before-new: known referents precede new ones
  - topic continuity: anaphoric references when topic persists
  - logical ordering: cause before effect for explanations,
    general before specific for definitions
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from strands.realization.render import FillerValue


class ActType(str, Enum):
    """Discourse-pragmatic categories. Drive ordering / coherence
    decisions in the planner; do not directly select templates
    (that's the leaf's ``template_shape``)."""
    # Single-act terminals
    ASSERT = "assert"           # informational claim
    VERIFY = "verify"           # yes/no verdict
    ACKNOWLEDGE = "acknowledge" # 'Got it'
    ELABORATE = "elaborate"     # follow-on fact about active topic
    GREET = "greet"
    THANK_RESPONSE = "thank_response"
    APOLOGIZE_RESPONSE = "apologize_response"
    FAREWELL = "farewell"
    INSTRUCT_RESPONSE = "instruct_response"
    DEFER = "defer"             # 'I don't know'
    REPHRASE_REQUEST = "rephrase_request"
    # Internal / compositional
    SEQUENCE = "sequence"       # ordered children
    CONTRAST = "contrast"       # 'X but Y'
    JUSTIFY = "justify"         # 'X because Y'


@dataclass(slots=True)
class ConceptualLeaf:
    """Terminal node in the response tree. Carries the conceptual
    content + a hint about which template to realize it with.

    Realization picks a template by (template_id | template_shape |
    template_relation), in that order, and renders it with
    ``fillers``. When ``flagged_for_compute`` is True the realizer
    delegates the whole leaf to the compute module instead of using
    a template.
    """
    fillers: dict[str, FillerValue | list[str] | None] = field(default_factory=dict)
    template_id: str = ""
    template_shape: str = ""
    template_relation: int = 0
    confidence: float = 1.0
    flagged_for_compute: bool = False
    intervention: str = ""
    backbone_node_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class CommunicativeAct:
    """Internal node OR a wrapper around a single leaf.

    Either ``children`` (composition) or ``leaf`` (terminal) is set.
    ``act_type`` informs ordering and discourse-state updates; the
    surface form is decided per leaf in §3.5.

    ``separator`` is the literal text inserted between this act and
    the next sibling — usually ' ' or ''. Lets the planner control
    whitespace without inflating the leaf templates.
    """
    act_type: ActType
    children: list["CommunicativeAct"] = field(default_factory=list)
    leaf: ConceptualLeaf | None = None
    separator: str = " "
    confidence: float = 1.0

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None

    def walk_leaves(self) -> Iterator[ConceptualLeaf]:
        """Depth-first left-to-right traversal of leaves."""
        if self.is_leaf:
            assert self.leaf is not None
            yield self.leaf
            return
        for c in self.children:
            yield from c.walk_leaves()


@dataclass(slots=True)
class ResponseStructure:
    """Full plan returned by the response planner. ``root`` is the
    top-level act; ``confidence`` is the overall planner confidence
    (typically the min over leaves)."""
    root: CommunicativeAct
    confidence: float = 1.0
    # Optional metadata for downstream consumers (training data,
    # debug). Not used by realization itself.
    intent: str = ""
    response_shape: str = ""

    def walk_leaves(self) -> Iterator[ConceptualLeaf]:
        return self.root.walk_leaves()

    @property
    def has_pending_compute(self) -> bool:
        """True if any leaf is fully delegated to compute, OR if a
        leaf's template will fail to render some non-COMPUTE slot
        (caught at realize time)."""
        return any(leaf.flagged_for_compute for leaf in self.walk_leaves())


# --- Shorthand constructors for the common single-leaf case -----------


def single_leaf(
    act_type: ActType,
    *,
    template_shape: str = "",
    template_relation: int = 0,
    template_id: str = "",
    fillers: dict[str, FillerValue | list[str] | None] | None = None,
    confidence: float = 1.0,
    intent: str = "",
    flagged_for_compute: bool = False,
    intervention: str = "",
    backbone_node_ids: tuple[int, ...] = (),
) -> ResponseStructure:
    """Build a ResponseStructure with a single act wrapping one leaf.
    The most common shape — every existing answerer fits this."""
    leaf = ConceptualLeaf(
        fillers=fillers or {},
        template_id=template_id,
        template_shape=template_shape,
        template_relation=template_relation,
        confidence=confidence,
        flagged_for_compute=flagged_for_compute,
        intervention=intervention,
        backbone_node_ids=backbone_node_ids,
    )
    root = CommunicativeAct(
        act_type=act_type, leaf=leaf, confidence=confidence,
    )
    return ResponseStructure(
        root=root, confidence=confidence, intent=intent,
        response_shape=template_shape,
    )


def sequence(
    children: list[CommunicativeAct],
    *,
    separator: str = " ",
    confidence: float | None = None,
) -> CommunicativeAct:
    """Compositional helper: a SEQUENCE act of multiple children.
    Used by planners that want to emit multiple sentences."""
    if confidence is None:
        confidence = (
            min(c.confidence for c in children) if children else 1.0
        )
    return CommunicativeAct(
        act_type=ActType.SEQUENCE,
        children=children,
        separator=separator,
        confidence=confidence,
    )
