"""Phrasal templates for surface realization (BDRM §2.4 + §3.5).

A ``Template`` is a parameterized linguistic pattern with named slots.
At realization time the model layer picks a template appropriate to
the response shape, fills its slots from the backbone (or defers a
slot to the compute module), and emits the result.

The slot mechanism is the core of the slot/compute interchange:
each slot is filled either deterministically (from a backbone lemma /
gloss / hypernym / etc.) or by the compute module. Both kinds of
fills are first-class, so the same template can produce a fully
deterministic answer or a hybrid answer depending on confidence.

Templates are stored separately from backbone nodes per spec §2.4
('Stored separately but linked from nodes') so the same template can
serve many concepts and templates can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SlotKind(str, Enum):
    """What kind of value goes in a slot, which drives both lookup
    and inflection rules during realization."""
    LEMMA = "lemma"            # a single word from a backbone node
    NOUN_PHRASE = "noun_phrase"  # may be multiword, may take an article
    LEMMA_LIST = "lemma_list"   # 'X and Y' / 'X, Y, and Z'
    GLOSS = "gloss"            # a synset definition
    LITERAL = "literal"        # a fixed string from the planner
    COMPUTE = "compute"        # filled by the compute module


class Register(str, Enum):
    NEUTRAL = "neutral"
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"


@dataclass(slots=True, frozen=True)
class SlotSpec:
    """Constraints on how a slot may be filled. Drives both the
    lookup (what concept_type is acceptable) and the inflection
    (whether to add an article, whether to pluralize)."""
    name: str
    kind: SlotKind
    article: bool = False           # add 'a/an' before the filler
    capitalize: bool = False        # uppercase first letter of filler
    plural: bool = False            # render as plural form
    # Compute-only: a hint to the NN about what kind of decision
    # this slot represents. Mirrors the spec's intervention spec.
    compute_intervention: str = ""  # 'reference' | 'factual' | 'creative' | …
    # Optional: when set, the realizer may try multiple senses of
    # the underlying lemma to find the right inflection.
    require_pos: str = ""           # 'noun' | 'verb' | 'adj' | ''


@dataclass(slots=True, frozen=True)
class Template:
    """A parameterized linguistic pattern.

    ``pattern`` uses ``{slot_name}`` placeholders that match the
    ``name`` field of one of the ``slots``. Literal text outside
    placeholders is emitted verbatim.

    Templates are tagged with at least one of:
      - ``response_shape``: a high-level category like 'definition',
        'yesno_yes', 'elaboration'. Matched by the response planner.
      - ``relation_type``: the Rel.* enum int when the template
        renders a single relation edge. Matched by relation walkers.

    A template can have both: e.g. an elaboration template tied to
    HYPERNYM. ``weight`` ranks competing templates; higher wins.
    ``register`` lets the planner pick by tone.
    """
    template_id: str               # stable id for refs / training
    pattern: str
    slots: tuple[SlotSpec, ...] = ()
    response_shape: str = ""
    relation_type: int = 0         # 0 = not relation-bound
    register: Register = Register.NEUTRAL
    weight: float = 1.0
    # When True the template is allowed even when one of its non-
    # COMPUTE slots can't be filled — the missing slot is converted
    # to a COMPUTE slot. Lets the system gracefully degrade.
    allow_compute_fallback: bool = True


@dataclass(slots=True)
class Segment:
    """A single contiguous chunk of rendered output. Each segment
    knows its source so streaming consumers can decide to emit it
    immediately (literals, lemmas) or pause for the compute module
    (compute-pending segments)."""
    text: str
    source: str                    # 'literal' | 'lemma' | 'gloss' |
                                   # 'lemma_list' | 'compute_pending' |
                                   # 'compute_resolved'
    confidence: float = 1.0
    backbone_node_id: int = -1
    # Set when source == 'compute_pending'. The realizer leaves these
    # slots open; the caller (or the compute module) fills them.
    pending_slot: SlotSpec | None = None
    pending_intervention: str = ""


@dataclass(slots=True)
class Rendered:
    """The output of rendering one template. A list of segments plus
    the chosen template id (for debugging / training) and overall
    confidence (min over deterministic segments).

    ``finalize()`` returns the flat string assuming all compute slots
    are resolved (or replacing them with their pending text)."""
    segments: list[Segment] = field(default_factory=list)
    template_id: str = ""
    response_shape: str = ""
    confidence: float = 1.0

    @property
    def has_pending_compute(self) -> bool:
        return any(s.source == "compute_pending" for s in self.segments)

    @property
    def pending_slots(self) -> list[Segment]:
        return [s for s in self.segments if s.source == "compute_pending"]

    def finalize(self) -> str:
        """Concatenate all segments. Compute-pending segments emit
        their (possibly placeholder) text; callers that want to fill
        them via the compute module should do so before calling."""
        return "".join(s.text for s in self.segments)
