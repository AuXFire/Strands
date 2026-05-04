"""End-to-end integration test (BDRM full system).

One long multi-turn conversation that exercises every layer of the
deterministic pipeline + the Compute Module seam:

  - definition Q&A (gloss-backed)
  - inform turns with belief extraction
  - elaboration walks (relation-walk for novel facts)
  - multi-turn coreference (pronoun resolution)
  - typed wh-questions (location/ability/composition)
  - yes/no questions with explicit negation
  - low-content middle turns that preserve the topic
  - low-confidence prompts that engage the Compute Module
  - rich Conditioning capture for training data

This is the test we point at when verifying the whole system still
works after a refactor — if it passes, the architecture is intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands.backbone import (
    DiscourseState,
    RecordingComputeModule,
    StubComputeModule,
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


def test_full_conversation_pipeline(backbone, tmp_path):
    """Drive a 12-turn conversation through the full pipeline and
    verify each layer behaves correctly."""
    out = tmp_path / "session.jsonl"
    state = DiscourseState()
    cm = RecordingComputeModule(out, inner=StubComputeModule(override=None))

    # Turn 1: definition question — gloss-backed, high confidence.
    r = respond(backbone, "What is a cat?", state=state, compute=cm)
    assert "cat" in r.text.lower()
    assert r.confidence >= 0.85
    assert r.compute_module_used is False

    # Turn 2: inform — should register belief and echo back.
    r = respond(backbone, "Cats are cute", state=state, compute=cm)
    assert "cats are cute" in r.text.lower()
    assert len(state.session_beliefs) == 1

    # Turn 3: another belief.
    r = respond(backbone, "Cats can purr", state=state, compute=cm)
    assert len(state.session_beliefs) == 2

    # Turn 4: elaboration — should walk relations to a novel fact.
    r1 = respond(backbone, "Tell me more.", state=state, compute=cm)
    assert "cat" in r1.text.lower()
    # Turn 5: another elaboration — must produce a DIFFERENT fact.
    r2 = respond(backbone, "And?", state=state, compute=cm)
    assert r1.text != r2.text

    # Turn 6: pronoun resolution — 'it' should resolve to cat.
    r = respond(backbone, "What is it?", state=state, compute=cm)
    assert r.pronoun_resolved is True
    assert "cat" in r.text.lower()

    # Turn 7: typed wh-question — location.
    r = respond(backbone, "Where do cats live?", state=state, compute=cm)
    assert "can be found at" in r.text.lower() or "cat" in r.text.lower()

    # Turn 8: yes/no with hypernym closure — YES.
    r = respond(backbone, "Is a cat an animal?", state=state, compute=cm)
    assert r.text.lower().startswith("yes"), r.text

    # Turn 9: yes/no NO with corrective category.
    r = respond(backbone, "Is a cat a dog?", state=state, compute=cm)
    assert r.text.lower().startswith("no"), r.text
    assert " is " in r.text  # the 'X is a Y' corrective clause

    # Turn 10: capability yes/no.
    r = respond(backbone, "Can a bird fly?", state=state, compute=cm)
    assert r.text.lower().startswith("yes"), r.text

    # Turn 11: social.
    r = respond(backbone, "Thanks for your help.", state=state, compute=cm)
    assert "welcome" in r.text.lower()

    # Turn 12: low-confidence — should flag CM and (since inner returned
    # None) keep the deterministic fallback.
    r = respond(backbone, "What is xyzzy?", state=state, compute=cm)
    assert r.needs_compute_module is True
    assert r.compute_module_used is False  # inner deferred
    assert "don't" in r.text.lower()

    cm.close()

    # Discourse state survived intact.
    assert state.turn_count == 12
    assert len(state.session_beliefs) == 2  # 'Cats are cute', 'Cats can purr'
    assert len(state.history) == 12

    # Captured training data — one example per Compute Module call,
    # which fires only on confidence < floor (0.5 default). At least
    # the final low-confidence turn should be captured.
    assert cm.count >= 1
    captured = [json.loads(l) for l in out.read_text().splitlines()]
    last = captured[-1]
    assert last["conditioning"]["prompt"] == "What is xyzzy?"
    # The training example must carry history + beliefs + speech_act.
    assert len(last["conditioning"]["history"]) > 0
    assert len(last["conditioning"]["user_beliefs"]) == 2
    assert last["conditioning"]["speech_act"] is not None


def test_conditioning_richness_sanity(backbone):
    """Spot check: the Conditioning the NN sees on a realistic prompt
    has all the structure we promised it would."""
    cm = StubComputeModule(override=None)
    state = DiscourseState()
    respond(backbone, "What is a bird?", state=state)
    respond(backbone, "Birds can fly", state=state)
    respond(
        backbone, "What is xyzzy?", state=state, compute=cm,
    )
    c = cm.calls[0]

    # History — the prior 2 turns are visible.
    assert len(c.history) == 2
    assert c.history[0].prompt == "What is a bird?"
    assert c.history[1].prompt == "Birds can fly"

    # User beliefs — the inform turn was lifted.
    assert len(c.user_beliefs) == 1
    assert c.user_beliefs[0].subject_lemma == "birds"
    assert c.user_beliefs[0].relation == "CAPABLE_OF"

    # Speech act — current turn is a question.
    assert c.speech_act is not None
    assert c.speech_act.act == "question"

    # Deterministic candidate present.
    assert c.deterministic_answer
    assert c.deterministic_confidence < 0.5

    # All fields are flat strings/numbers — JSON-friendly.
    from strands.backbone import conditioning_to_dict
    d = conditioning_to_dict(c)
    json.dumps(d)  # must not raise


def test_pipeline_remains_deterministic(backbone):
    """Same conversation twice should produce identical responses
    and identical Conditioning payloads — no randomness anywhere."""
    def run() -> tuple[list[str], list[int]]:
        state = DiscourseState()
        responses: list[str] = []
        anchors: list[int] = []
        for p in [
            "What is a cat?",
            "Cats are cute",
            "Tell me more.",
            "What is it?",
            "Is a cat an animal?",
            "Where do cats live?",
        ]:
            r = respond(backbone, p, state=state)
            responses.append(r.text)
            anchors.append(r.primary_anchor_id or -1)
        return responses, anchors

    a1, n1 = run()
    a2, n2 = run()
    assert a1 == a2
    assert n1 == n2
