"""Slot filling and rendering (BDRM §3.5).

Given a Template and a dict of slot fillers, walk the template
pattern emitting Segments. Literal text becomes 'literal' segments;
named placeholders look up the slot's filler, apply inflection, and
become 'lemma' / 'gloss' / 'lemma_list' / 'literal' / 'compute_pending'
segments depending on the SlotKind and whether the filler is provided.

This is the seam where deterministic and compute fills become
interchangeable: an unfilled non-COMPUTE slot can degrade to a
'compute_pending' segment so the realization completes (and the
caller decides whether to invoke the compute module to fill it).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from strands.realization.template import (
    Rendered,
    Segment,
    SlotKind,
    SlotSpec,
    Template,
)


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_VOWELS = frozenset("aeiouAEIOU")


def _article_for(word: str) -> str:
    """English a/an by initial vowel sound. Heuristic only."""
    if not word:
        return "a"
    return "an" if word[0] in _VOWELS else "a"


def _apply_article(text: str, *, add_article: bool) -> str:
    if not add_article or not text:
        return text
    return f"{_article_for(text)} {text}"


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


@dataclass(slots=True)
class FillerValue:
    """The filler the planner supplies for one slot.

    ``value`` is the raw text (lemma, gloss, etc.).
    ``source`` records where it came from for the resulting Segment.
    ``backbone_node_id`` lets segments trace back to backbone state.
    ``confidence`` gates the audit and the compute-fallback decision.
    """
    value: str
    source: str = "lemma"           # one of Segment.source values
    backbone_node_id: int = -1
    confidence: float = 1.0


def _render_lemma_list(values: list[str]) -> str:
    """English coordination: 'X', 'X and Y', 'X, Y, and Z'."""
    cleaned = [v for v in values if v]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _resolve_slot_text(
    spec: SlotSpec, filler: FillerValue | list[str] | None,
) -> str | None:
    """Convert a raw filler into the string the template emits for
    that slot, applying article / capitalize / pluralize. Returns
    None when the slot can't be filled deterministically."""
    if filler is None:
        return None
    if spec.kind == SlotKind.LEMMA_LIST:
        if isinstance(filler, FillerValue):
            # Allow planners to pass a single FillerValue with comma-
            # separated content for convenience.
            text = filler.value
        elif isinstance(filler, list):
            text = _render_lemma_list(filler)
        else:
            return None
        if not text:
            return None
        if spec.capitalize:
            text = _capitalize_first(text)
        return text

    # All other kinds: a single FillerValue.
    if not isinstance(filler, FillerValue):
        return None
    text = filler.value
    if not text:
        return None
    if spec.kind == SlotKind.GLOSS:
        # Glosses are usually lowercase fragments; lowercase the first
        # char so the surrounding sentence reads naturally.
        text = text[0].lower() + text[1:] if len(text) > 1 else text
    if spec.article:
        text = _apply_article(text, add_article=True)
    if spec.capitalize:
        text = _capitalize_first(text)
    return text


def _segment_for_slot(
    spec: SlotSpec, filler: FillerValue | list[str] | None,
    *, allow_compute_fallback: bool,
) -> Segment:
    """Build the segment for one slot, including the compute-fallback
    path when the slot can't be filled deterministically."""
    # Explicit COMPUTE slot: always compute-pending, regardless of
    # whether the planner supplied something.
    if spec.kind == SlotKind.COMPUTE:
        return Segment(
            text=f"⟨{spec.name}⟩",       # placeholder text for streams
            source="compute_pending",
            confidence=0.0,
            pending_slot=spec,
            pending_intervention=spec.compute_intervention,
        )

    text = _resolve_slot_text(spec, filler)
    if text is None:
        if not allow_compute_fallback:
            # No fallback allowed — emit empty literal so the caller
            # can detect failure.
            return Segment(text="", source="literal", confidence=0.0)
        return Segment(
            text=f"⟨{spec.name}⟩",
            source="compute_pending",
            confidence=0.0,
            pending_slot=spec,
            pending_intervention=spec.compute_intervention,
        )

    source = {
        SlotKind.LEMMA: "lemma",
        SlotKind.NOUN_PHRASE: "lemma",
        SlotKind.LEMMA_LIST: "lemma_list",
        SlotKind.GLOSS: "gloss",
        SlotKind.LITERAL: "literal",
    }.get(spec.kind, "lemma")

    if isinstance(filler, FillerValue):
        return Segment(
            text=text,
            source=source,
            confidence=filler.confidence,
            backbone_node_id=filler.backbone_node_id,
        )
    return Segment(text=text, source=source)


def render(
    template: Template,
    fillers: Mapping[str, FillerValue | list[str] | None] | None = None,
) -> Rendered:
    """Render a template with the supplied slot fillers. Missing
    non-COMPUTE slots become compute_pending segments unless the
    template forbids it.

    ``fillers`` keys are slot names. Values can be FillerValue (single)
    or list[str] (for LEMMA_LIST slots) or None to force compute.
    """
    fillers = fillers or {}
    slot_by_name = {s.name: s for s in template.slots}
    rendered = Rendered(
        template_id=template.template_id,
        response_shape=template.response_shape,
    )

    pos = 0
    for m in _PLACEHOLDER_RE.finditer(template.pattern):
        start, end = m.span()
        if start > pos:
            literal = template.pattern[pos:start]
            if literal:
                rendered.segments.append(
                    Segment(text=literal, source="literal"),
                )
        slot_name = m.group(1)
        spec = slot_by_name.get(slot_name)
        if spec is None:
            # Unknown placeholder — emit literal as-is so we don't
            # silently swallow errors.
            rendered.segments.append(
                Segment(text=m.group(0), source="literal", confidence=0.0),
            )
        else:
            seg = _segment_for_slot(
                spec, fillers.get(slot_name),
                allow_compute_fallback=template.allow_compute_fallback,
            )
            rendered.segments.append(seg)
        pos = end
    if pos < len(template.pattern):
        tail = template.pattern[pos:]
        rendered.segments.append(Segment(text=tail, source="literal"))

    # Overall confidence = min over deterministic segments.
    det_segs = [
        s for s in rendered.segments
        if s.source in {"lemma", "gloss", "lemma_list", "literal"}
        and s.confidence > 0
    ]
    rendered.confidence = (
        min(s.confidence for s in det_segs) if det_segs else 1.0
    )
    return rendered


def render_text(
    template: Template,
    fillers: Mapping[str, FillerValue | list[str] | None] | None = None,
) -> str:
    """Convenience: render and return the flat string."""
    return render(template, fillers).finalize()
