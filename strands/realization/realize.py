"""Realize a ResponseStructure into surface text (BDRM §3.5).

Walks the tree depth-first, left-to-right. For each leaf, applies
the four-way decision from §3.5:

  1. ``leaf.flagged_for_compute`` → emit a single compute_pending
     segment (whole leaf delegated to the NN)
  2. Template by id / shape / relation → render with fillers; missing
     non-COMPUTE slots become compute_pending segments inside the
     template (slot-level compute, not leaf-level)
  3. (Reserved for B3) single-lemma emission with inflection
  4. (Reserved for B3) primitive composition via assembly rules

For B2 the realizer focuses on (1) and (2). The lemma-only and
composed paths are filled in during B3 once the planner emits leaves
that need them.

``realize`` returns a ``RealizedResponse`` carrying:
  - ``text``: the flat string with placeholders for any pending
    compute segments
  - ``confidence``: min over deterministic leaf confidences
  - ``segments``: the full per-leaf segment stream so a streaming
    consumer can pause for compute
  - ``has_pending_compute``: True if any segment needs compute
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strands.realization.render import render
from strands.realization.store import TemplateStore
from strands.realization.structure import (
    CommunicativeAct,
    ConceptualLeaf,
    ResponseStructure,
)
from strands.realization.template import Segment


@dataclass(slots=True)
class RealizedResponse:
    """Output of realize(). Mirrors Rendered but at the tree level."""
    text: str
    confidence: float = 1.0
    segments: list[Segment] = field(default_factory=list)

    @property
    def has_pending_compute(self) -> bool:
        return any(s.source == "compute_pending" for s in self.segments)

    @property
    def pending_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.source == "compute_pending"]


def _resolve_template_for_leaf(
    leaf: ConceptualLeaf, store: TemplateStore,
):
    """Pick a template for this leaf. Priority: id → shape×relation →
    shape → relation. Returns None if nothing matches."""
    if leaf.template_id:
        t = store.get(leaf.template_id)
        if t is not None:
            return t
    if leaf.template_shape and leaf.template_relation:
        t = store.best(
            shape=leaf.template_shape,
            relation_type=leaf.template_relation,
        )
        if t is not None:
            return t
    if leaf.template_shape:
        t = store.best(shape=leaf.template_shape)
        if t is not None:
            return t
    if leaf.template_relation:
        t = store.best(relation_type=leaf.template_relation)
        if t is not None:
            return t
    return None


def _realize_leaf(
    leaf: ConceptualLeaf, store: TemplateStore,
) -> tuple[list[Segment], float]:
    """Realize a single leaf to segments. Confidence is the leaf's
    own confidence floored by the rendered template's min-segment
    confidence."""
    if leaf.flagged_for_compute:
        # Whole-leaf delegation: one compute_pending segment.
        seg = Segment(
            text=f"⟨{leaf.intervention or 'compute'}⟩",
            source="compute_pending",
            confidence=0.0,
            pending_intervention=leaf.intervention,
        )
        return [seg], 0.0

    template = _resolve_template_for_leaf(leaf, store)
    if template is None:
        # No template found — this leaf can't be rendered
        # deterministically. Emit a compute_pending segment so the
        # caller sees the gap.
        seg = Segment(
            text=f"⟨{leaf.template_shape or leaf.template_id or '???'}⟩",
            source="compute_pending",
            confidence=0.0,
            pending_intervention=leaf.intervention or "fill_template",
        )
        return [seg], 0.0

    rendered = render(template, leaf.fillers)
    return rendered.segments, min(leaf.confidence, rendered.confidence)


def _walk_act(
    act: CommunicativeAct, store: TemplateStore,
) -> tuple[list[Segment], float]:
    """Recursively walk an act; return its accumulated segments and
    confidence."""
    if act.is_leaf:
        assert act.leaf is not None
        segs, conf = _realize_leaf(act.leaf, store)
        return segs, conf

    all_segs: list[Segment] = []
    all_confs: list[float] = []
    for i, child in enumerate(act.children):
        if i > 0 and act.separator:
            all_segs.append(
                Segment(text=act.separator, source="literal"),
            )
        child_segs, child_conf = _walk_act(child, store)
        all_segs.extend(child_segs)
        all_confs.append(child_conf)
    conf = min(all_confs) if all_confs else 1.0
    return all_segs, conf


def realize(
    structure: ResponseStructure,
    store: TemplateStore,
) -> RealizedResponse:
    """Tree-walk a ResponseStructure into surface output via the
    template store. Compute-pending segments stay open; the caller
    decides whether to invoke the compute module to fill them."""
    segments, conf = _walk_act(structure.root, store)
    text = "".join(s.text for s in segments)
    return RealizedResponse(
        text=text,
        confidence=min(conf, structure.confidence),
        segments=segments,
    )
