"""Compute Module integration seam tests (BDRM §4).

Verifies the contract between the deterministic pipeline and an injected
Compute Module: it fires only on low-confidence outputs, the conditioning
payload carries everything a NN needs, and a ``None`` return defers to
the deterministic answer.

Uses ``StubComputeModule`` — no real NN required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    AnchorFact,
    Conditioning,
    StubComputeModule,
    build_conditioning,
    infer,
    load,
    respond,
)

_BACKBONE_DIR = Path(__file__).parent.parent / "strands" / "data" / "backbone"


@pytest.fixture(scope="module")
def backbone():
    if not (_BACKBONE_DIR / "backbone.manifest.json").exists():
        pytest.skip(
            "backbone not built — run `python scripts/build_backbone.py`"
        )
    return load(_BACKBONE_DIR)


def test_compute_module_fires_on_low_confidence(backbone):
    """An unknown-word question hits the low-confidence path; the
    Compute Module override should replace the deterministic answer."""
    cm = StubComputeModule(override="[NN-generated answer]")
    r = respond(backbone, "What is xyzzy?", compute=cm)
    assert r.needs_compute_module is True
    assert r.compute_module_used is True
    assert r.text == "[NN-generated answer]"
    assert len(cm.calls) == 1


def test_compute_module_skipped_on_high_confidence(backbone):
    """Gloss-backed Q&A is high confidence — the Compute Module must
    not be invoked at all."""
    cm = StubComputeModule(override="[NN should not be called]")
    r = respond(backbone, "What is photosynthesis?", compute=cm)
    assert r.needs_compute_module is False
    assert r.compute_module_used is False
    assert "[NN" not in r.text
    assert len(cm.calls) == 0


def test_compute_module_defer_keeps_deterministic(backbone):
    """When the Compute Module returns None, the deterministic answer
    is preserved even though the seam was exercised."""
    cm = StubComputeModule(override=None)
    r = respond(backbone, "What is xyzzy?", compute=cm)
    assert r.needs_compute_module is True
    assert r.compute_module_used is False  # CM ran but deferred
    assert "I don't" in r.text  # deterministic fallback wording
    assert len(cm.calls) == 1


def test_no_compute_module_attached(backbone):
    """If no Compute Module is supplied, low-confidence responses
    flag the need but keep the deterministic answer."""
    r = respond(backbone, "What is xyzzy?")
    assert r.needs_compute_module is True
    assert r.compute_module_used is False
    assert r.text  # non-empty fallback


def test_conditioning_payload_carries_state(backbone):
    """The Conditioning handed to the CM should expose everything
    needed for grounded generation: prompt, intent, primary anchor
    (with lemmas + gloss + hypernym), related anchors, and the
    deterministic candidate answer."""
    cm = StubComputeModule(override="ok")
    respond(backbone, "What is xyzzy?", compute=cm)
    assert len(cm.calls) == 1
    c = cm.calls[0]
    assert isinstance(c, Conditioning)
    assert c.prompt == "What is xyzzy?"
    assert c.intent == "question_answering"
    assert "xyzzy" in c.unknowns
    assert c.deterministic_answer  # whatever the deterministic fallback was
    assert c.deterministic_confidence < 0.5


def test_conditioning_with_known_anchor(backbone):
    """When the prompt has known anchors, primary_anchor should be
    populated with lemmas, gloss, and hypernym lemma."""
    # Use a lower confidence floor to force the CM seam to fire even
    # on a high-confidence answer, so we can inspect the payload.
    cm = StubComputeModule(override=None)  # defer — keep deterministic
    respond(
        backbone, "What is photosynthesis?",
        compute=cm, confidence_floor=0.99,
    )
    assert len(cm.calls) == 1
    c = cm.calls[0]
    assert c.primary_anchor is not None
    assert isinstance(c.primary_anchor, AnchorFact)
    assert any("photosynthesis" in l for l in c.primary_anchor.lemmas)
    assert c.primary_anchor.gloss  # gloss table populated
    assert c.primary_anchor.activation > 0.0


def test_confidence_floor_override(backbone):
    """A high confidence_floor forces the CM seam on every prompt,
    letting callers route everything through the NN if they want."""
    cm = StubComputeModule(override="forced override")
    r = respond(
        backbone, "What is photosynthesis?",
        compute=cm, confidence_floor=0.99,
    )
    assert r.needs_compute_module is True
    assert r.compute_module_used is True
    assert r.text == "forced override"


def test_build_conditioning_directly(backbone):
    """``build_conditioning`` is callable independently — useful for
    test harnesses and the future training pipeline."""
    result = infer(backbone, "The dog chased the cat")
    # Pick the first anchor for testing.
    primary = next(iter(result.anchors.values()))
    c = build_conditioning(
        backbone, result,
        primary_anchor_id=primary,
        deterministic_answer="placeholder",
        deterministic_confidence=0.4,
        related_top_k=3,
    )
    assert c.prompt == "The dog chased the cat"
    assert c.primary_anchor is not None
    assert c.primary_anchor.node_id == primary
    assert len(c.related_anchors) <= 3
    # Related anchors should not include the primary.
    assert all(r.node_id != primary for r in c.related_anchors)
    # Activations are sorted descending in related list.
    if len(c.related_anchors) >= 2:
        assert c.related_anchors[0].activation >= c.related_anchors[1].activation
