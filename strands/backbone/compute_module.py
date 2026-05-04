"""Compute Module integration point (BDRM §4).

The Compute Module is the optional neural fallback that fires when the
deterministic pipeline (backbone + model layer + template realization)
returns a low-confidence response. The spec sizes it at 200-500M params,
GPU-resident, sparse-activated.

This module defines only the *seam*: a ``Conditioning`` payload built
from the deterministic state, a ``ComputeModule`` protocol, and a
``StubComputeModule`` that lets us test the integration without a
trained network.

A real implementation lives in ``strands.compute.*`` (out of scope here)
and is injected at runtime via ``respond(..., compute=...)``.

The contract is intentionally narrow:
  - input  = whatever the deterministic path already produced
            (prompt, intent, anchors, top-activated subgraph, glosses,
            deterministic candidate answer, confidence)
  - output = a replacement string, or ``None`` to keep the deterministic
            answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from strands.backbone.inference import InferenceResult
from strands.backbone.loader import Backbone

if TYPE_CHECKING:
    from strands.backbone.response import TurnRecord


@dataclass(slots=True)
class Fact:
    """One typed edge from the subject anchor, pre-rendered for the
    Compute Module. ``relation`` is a human-readable string drawn from
    the Rel enum (e.g. 'HYPERNYM'); ``weight`` is the normalized edge
    weight in [0, 1]."""
    relation: str
    target_lemma: str
    weight: float = 0.0


@dataclass(slots=True)
class AnchorFact:
    """One anchor packaged for the Compute Module: lemmas + gloss +
    (optional) hypernym lemma + first-hop facts. Pre-rendered so the
    NN sees clean text rather than node IDs."""
    node_id: int
    lemmas: list[str]
    gloss: str
    hypernym_lemma: str = ""
    activation: float = 0.0
    facts: list[Fact] = field(default_factory=list)


@dataclass(slots=True)
class HistoryTurn:
    """One past turn rendered for the Compute Module. A flattening of
    response.TurnRecord that drops backbone IDs the NN doesn't need."""
    turn_index: int
    prompt: str
    response: str
    intent: str
    question_type: str = ""
    anchor_lemma: str = ""
    pronoun_resolved: bool = False


@dataclass(slots=True)
class Conditioning:
    """The full conditioning payload handed to the Compute Module.

    Everything the NN needs to produce a grounded response is here.
    No raw backbone access — all node IDs have already been resolved
    to lemmas/glosses by the model layer.
    """
    prompt: str
    intent: str
    primary_anchor: AnchorFact | None
    related_anchors: list[AnchorFact] = field(default_factory=list)
    deterministic_answer: str = ""
    deterministic_confidence: float = 0.0
    unknowns: list[str] = field(default_factory=list)
    history: list[HistoryTurn] = field(default_factory=list)


class ComputeModule(Protocol):
    """Protocol any Compute Module implementation must satisfy.

    ``complete`` returns ``None`` to defer to the deterministic answer,
    or a string to override it. Implementations must be side-effect free
    on the backbone — they read the conditioning, they don't mutate it.
    """

    def complete(self, conditioning: Conditioning) -> str | None: ...


# --- Stub implementation -----------------------------------------------


class StubComputeModule:
    """A no-op Compute Module used in tests and as the default for
    callers who haven't wired in a real NN. Records every call it
    receives so tests can assert the seam fired.

    Returns ``None`` (defer to deterministic) by default. Set
    ``override`` to a string to force a replacement, simulating an
    NN that took control."""

    def __init__(self, override: str | None = None) -> None:
        self.override = override
        self.calls: list[Conditioning] = []

    def complete(self, conditioning: Conditioning) -> str | None:
        self.calls.append(conditioning)
        return self.override


# --- Conditioning builder ---------------------------------------------


def _best_lemma(backbone: Backbone, node_id: int) -> str:
    lemmas = backbone.lemmas_for(node_id)
    if not lemmas:
        return ""
    single = [l for l in lemmas if " " not in l]
    return (sorted(single, key=len)[0] if single else lemmas[0])


def _hypernym_lemma(backbone: Backbone, node_id: int) -> str:
    """Best-weight HYPERNYM lemma for the node, or '' if none."""
    from strands.backbone.schema import Rel  # local import — avoid cycle
    edges = backbone.edges_with_relation(node_id, Rel.HYPERNYM)
    if edges.size == 0:
        return ""
    best_idx = int(edges["weight"].argmax())
    return _best_lemma(backbone, int(edges[best_idx]["target_id"]))


# Relations rendered as facts on each AnchorFact, in priority order.
# The NN sees a flat list of (relation, target_lemma, weight) triples
# so it can ground generation in concrete edges.
_FACT_RELATIONS: tuple[int, ...] = (
    0x0001,  # HYPERNYM
    0x0002,  # HYPONYM
    0x0003,  # MERONYM
    0x0020,  # HAS_PROPERTY
    0x0022,  # CAPABLE_OF
    0x0030,  # AT_LOCATION
    0x0050,  # USED_FOR
    0x00E0,  # MADE_OF
    0x0010,  # CAUSES
    0x0011,  # CAUSED_BY
)


def _facts_for_node(
    backbone: Backbone, node_id: int, *, top_k_per_relation: int = 3,
) -> list[Fact]:
    """Pull up to ``top_k_per_relation`` highest-weight edges per
    relation type, rendered as flat Fact triples for the NN."""
    import numpy as np
    from strands.backbone.schema import Rel
    out: list[Fact] = []
    for rel_id in _FACT_RELATIONS:
        rel = Rel(rel_id)
        edges = backbone.edges_with_relation(node_id, rel)
        if edges.size == 0:
            continue
        order = np.argsort(-edges["weight"].astype(np.int64))
        for idx in order[:top_k_per_relation]:
            target_id = int(edges[int(idx)]["target_id"])
            target_lemma = _best_lemma(backbone, target_id)
            if not target_lemma:
                continue
            weight = float(edges[int(idx)]["weight"]) / 0xFFFF
            out.append(Fact(
                relation=rel.name,
                target_lemma=target_lemma,
                weight=weight,
            ))
    return out


def _anchor_fact(
    backbone: Backbone, node_id: int, activation: float,
    *, with_facts: bool = True,
) -> AnchorFact:
    return AnchorFact(
        node_id=node_id,
        lemmas=list(backbone.lemmas_for(node_id)),
        gloss=backbone.gloss_for(node_id),
        hypernym_lemma=_hypernym_lemma(backbone, node_id),
        activation=activation,
        facts=_facts_for_node(backbone, node_id) if with_facts else [],
    )


def _history_to_turns(history: "list | None") -> list[HistoryTurn]:
    """Flatten a list of TurnRecord into HistoryTurn (drops backbone IDs
    the NN doesn't need)."""
    if not history:
        return []
    out: list[HistoryTurn] = []
    for r in history:
        out.append(HistoryTurn(
            turn_index=r.turn_index,
            prompt=r.prompt,
            response=r.response,
            intent=r.intent,
            question_type=r.question_type,
            anchor_lemma=r.primary_anchor_lemma,
            pronoun_resolved=r.pronoun_resolved,
        ))
    return out


def build_conditioning(
    backbone: Backbone,
    inference: InferenceResult,
    *,
    primary_anchor_id: int | None,
    deterministic_answer: str,
    deterministic_confidence: float,
    related_top_k: int = 5,
    history: "list | None" = None,
) -> Conditioning:
    """Pack the deterministic state into a Conditioning payload.

    ``history`` is an optional list of TurnRecord (from DiscourseState)
    representing prior turns. When supplied, it's flattened into
    HistoryTurn entries so the Compute Module sees the conversation
    so far without backbone-internal IDs."""
    primary = (
        _anchor_fact(
            backbone,
            primary_anchor_id,
            inference.activations.get(primary_anchor_id, 0.0),
        )
        if primary_anchor_id is not None
        else None
    )

    # Related = highest-activation subgraph nodes other than the primary.
    # Skip facts on related anchors — keeps the payload bounded; the
    # primary anchor is the one that matters most for grounding.
    sorted_active = sorted(
        inference.activations.items(), key=lambda x: -x[1],
    )
    related: list[AnchorFact] = []
    for nid, act in sorted_active:
        if nid == primary_anchor_id:
            continue
        related.append(_anchor_fact(backbone, nid, act, with_facts=False))
        if len(related) >= related_top_k:
            break

    return Conditioning(
        prompt=inference.prompt,
        intent=inference.intent,
        primary_anchor=primary,
        related_anchors=related,
        history=_history_to_turns(history),
        deterministic_answer=deterministic_answer,
        deterministic_confidence=deterministic_confidence,
        unknowns=list(inference.unknowns),
    )
