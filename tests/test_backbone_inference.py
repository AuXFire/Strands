"""Prompt-to-backbone inference tests (BDRM §3.2).

Uses the production backbone artifacts under
strands/data/backbone/ — built once via scripts/build_backbone.py
and cached. Skips if absent so the test suite stays runnable on
fresh checkouts that haven't built the backbone yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    InferenceConfig,
    infer,
    load,
)

_BACKBONE_DIR = Path(__file__).parent.parent / "strands" / "data" / "backbone"


@pytest.fixture(scope="module")
def backbone():
    if not (_BACKBONE_DIR / "backbone.manifest.json").exists():
        pytest.skip(
            "backbone not built — run `python scripts/build_backbone.py`"
        )
    return load(_BACKBONE_DIR)


def _word_at_anchor(b, result, surface):
    """Return the lemma tuple of the node chosen for ``surface``."""
    for i, tok in enumerate(result.tokens):
        if tok.surface == surface and i in result.anchors:
            return b.lemmas_for(result.anchors[i])
    return None


def test_disambiguation_dog_animal_vs_firedog(backbone):
    """Same word, different contexts, two different senses chosen.
    This is the headline correctness test for graph-coherence WSD."""
    animal = infer(backbone, "The dog chased the cat")
    firedog = infer(backbone, "The dog iron held the firewood")

    animal_dog = _word_at_anchor(backbone, animal, "dog")
    firedog_dog = _word_at_anchor(backbone, firedog, "dog")

    assert animal_dog is not None
    assert firedog_dog is not None
    # They must resolve to different nodes — different senses.
    animal_node = animal.anchors[
        next(i for i, t in enumerate(animal.tokens) if t.surface == "dog")
    ]
    firedog_node = firedog.anchors[
        next(i for i, t in enumerate(firedog.tokens) if t.surface == "dog")
    ]
    assert animal_node != firedog_node, (
        f"disambiguation failed: dog resolved to same node {animal_node} "
        "in both contexts"
    )

    # Animal sense should mention 'canis' or 'domestic dog'; firedog
    # sense should mention 'andiron' or 'dog-iron'.
    assert any("canis" in l or "domestic" in l for l in animal_dog), (
        f"expected animal-sense lemmas, got {animal_dog}"
    )
    assert any("andiron" in l or "iron" in l for l in firedog_dog), (
        f"expected firedog-sense lemmas, got {firedog_dog}"
    )


def test_intent_question(backbone):
    r = infer(backbone, "How does photosynthesis work?")
    assert r.intent == "question_answering"


def test_intent_instruction(backbone):
    r = infer(backbone, "Please show me the weather forecast")
    assert r.intent == "instruction"


def test_intent_social(backbone):
    r = infer(backbone, "Hello there")
    assert r.intent in {"social", "inform"}  # depends on exact phrasing


def test_anchors_returned_for_known_words(backbone):
    r = infer(backbone, "The dog chased the cat")
    assert "dog" in {t.surface for t in r.tokens}
    assert "cat" in {t.surface for t in r.tokens}
    assert len(r.anchors) >= 2


def test_unknowns_for_garbage(backbone):
    r = infer(backbone, "xyzzy plugh frobozz")
    assert len(r.unknowns) == 3


def test_subgraph_grows_with_hop_limit(backbone):
    cfg1 = InferenceConfig(hop_limit=1)
    cfg2 = InferenceConfig(hop_limit=2)
    cfg3 = InferenceConfig(hop_limit=3)
    p = "The dog chased the cat"
    s1 = len(infer(backbone, p, cfg=cfg1).subgraph_node_ids)
    s2 = len(infer(backbone, p, cfg=cfg2).subgraph_node_ids)
    s3 = len(infer(backbone, p, cfg=cfg3).subgraph_node_ids)
    # Larger hop limits should not shrink the subgraph.
    assert s2 >= s1
    assert s3 >= s2


def test_anchors_have_high_activation(backbone):
    r = infer(backbone, "The dog chased the cat")
    for token_idx, node_id in r.anchors.items():
        assert r.activations.get(node_id, 0.0) > 0.0, (
            f"anchor {node_id} has zero activation"
        )


def test_known_pair_share_subgraph(backbone):
    """money ↔ bank: their subgraphs should overlap because of
    ConceptNet edges — money relates_to bank, bank has_a money etc."""
    r = infer(backbone, "money in the bank")
    assert len(r.subgraph_node_ids) > 5
    assert len(r.anchors) >= 2


def test_pure_deterministic(backbone):
    """No randomness, no NN — same prompt, same result."""
    p = "The dog chased the cat"
    r1 = infer(backbone, p)
    r2 = infer(backbone, p)
    assert r1.anchors == r2.anchors
    assert r1.intent == r2.intent
    assert r1.subgraph_node_ids == r2.subgraph_node_ids
