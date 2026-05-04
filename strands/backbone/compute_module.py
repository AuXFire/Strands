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
from typing import Protocol

from strands.backbone.inference import InferenceResult
from strands.backbone.loader import Backbone


@dataclass(slots=True)
class AnchorFact:
    """One anchor packaged for the Compute Module: lemmas + gloss +
    (optional) hypernym lemma. Pre-rendered so the NN sees clean text
    rather than node IDs."""
    node_id: int
    lemmas: list[str]
    gloss: str
    hypernym_lemma: str = ""
    activation: float = 0.0


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


def _hypernym_lemma(backbone: Backbone, node_id: int) -> str:
    """Best-weight HYPERNYM lemma for the node, or '' if none."""
    from strands.backbone.schema import Rel  # local import — avoid cycle
    edges = backbone.edges_with_relation(node_id, Rel.HYPERNYM)
    if edges.size == 0:
        return ""
    best_idx = int(edges["weight"].argmax())
    target = int(edges[best_idx]["target_id"])
    lemmas = backbone.lemmas_for(target)
    if not lemmas:
        return ""
    single = [l for l in lemmas if " " not in l]
    return (sorted(single, key=len)[0] if single else lemmas[0])


def _anchor_fact(
    backbone: Backbone, node_id: int, activation: float,
) -> AnchorFact:
    return AnchorFact(
        node_id=node_id,
        lemmas=list(backbone.lemmas_for(node_id)),
        gloss=backbone.gloss_for(node_id),
        hypernym_lemma=_hypernym_lemma(backbone, node_id),
        activation=activation,
    )


def build_conditioning(
    backbone: Backbone,
    inference: InferenceResult,
    *,
    primary_anchor_id: int | None,
    deterministic_answer: str,
    deterministic_confidence: float,
    related_top_k: int = 5,
) -> Conditioning:
    """Pack the deterministic state into a Conditioning payload."""
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
    sorted_active = sorted(
        inference.activations.items(), key=lambda x: -x[1],
    )
    related: list[AnchorFact] = []
    for nid, act in sorted_active:
        if nid == primary_anchor_id:
            continue
        related.append(_anchor_fact(backbone, nid, act))
        if len(related) >= related_top_k:
            break

    return Conditioning(
        prompt=inference.prompt,
        intent=inference.intent,
        primary_anchor=primary,
        related_anchors=related,
        deterministic_answer=deterministic_answer,
        deterministic_confidence=deterministic_confidence,
        unknowns=list(inference.unknowns),
    )
