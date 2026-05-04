"""End-to-end response generation tests (BDRM §3.4 + §3.5).

Covers the minimal Q&A path: prompt → inference → anchor → content
selection → surface realization → ``Response``. Uses the production
backbone artifacts, skipping if absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    DiscourseState,
    Response,
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


def test_question_returns_gloss_backed_answer(backbone):
    """A 'what is X' question should produce a sentence containing the
    subject and a definition drawn from the synset gloss."""
    r = respond(backbone, "What is photosynthesis?")
    assert isinstance(r, Response)
    assert r.inference.intent == "question_answering"
    assert r.text.lower().startswith("photosynthesis"), r.text
    # Gloss-backed answers carry high confidence.
    assert r.confidence >= 0.85
    assert r.needs_compute_module is False
    # Body should look definitional (some content past the subject).
    assert len(r.text) > len("Photosynthesis is .")


def test_question_unknown_word_low_confidence(backbone):
    """A question about a nonsense word has no anchor — confidence
    should be low and the Compute Module should be flagged."""
    r = respond(backbone, "What is xyzzy?")
    assert r.inference.intent == "question_answering"
    assert r.primary_anchor_id is None
    assert r.confidence < 0.5
    assert r.needs_compute_module is True


def test_social_greeting(backbone):
    r = respond(backbone, "Hello there")
    assert r.inference.intent in {"social", "inform"}
    # Greetings always render as a greeting.
    assert r.text.lower().startswith("hello") or "hi" in r.text.lower()


def test_social_thanks(backbone):
    r = respond(backbone, "Thanks for your help")
    assert "welcome" in r.text.lower()


def test_instruction_dispatch(backbone):
    r = respond(backbone, "Please show me the weather forecast")
    assert r.inference.intent == "instruction"
    # Instruction template starts with OK acknowledgement.
    assert r.text.lower().startswith("ok"), r.text


def test_inform_dispatch(backbone):
    r = respond(backbone, "The dog chased the cat")
    # 'The dog chased the cat' is declarative — should classify as inform.
    assert r.inference.intent in {"inform", "question_answering"}
    # In either case, we get a non-empty text.
    assert r.text


def test_deterministic(backbone):
    """Same prompt → byte-identical response (no NN, no randomness)."""
    p = "What is a cat?"
    r1 = respond(backbone, p)
    r2 = respond(backbone, p)
    assert r1.text == r2.text
    assert r1.confidence == r2.confidence
    assert r1.primary_anchor_id == r2.primary_anchor_id
    assert r1.inference.intent == r2.inference.intent


def test_discourse_state_tracks_turns(backbone):
    """DiscourseState should accumulate across turns when reused."""
    state = DiscourseState()
    assert state.turn_count == 0

    respond(backbone, "What is a cat?", state=state)
    assert state.turn_count == 1
    assert state.last_intent == "question_answering"
    assert len(state.active_topic_node_ids) > 0

    respond(backbone, "Hello", state=state)
    assert state.turn_count == 2


def test_discourse_state_default_fresh_each_call(backbone):
    """Without a passed-in state, a fresh DiscourseState is allocated."""
    r1 = respond(backbone, "What is a cat?")
    r2 = respond(backbone, "What is a cat?")
    assert r1.state is not None
    assert r2.state is not None
    # Different instances, both showing turn_count==1.
    assert r1.state is not r2.state
    assert r1.state.turn_count == 1
    assert r2.state.turn_count == 1


def test_response_carries_inference_result(backbone):
    """The Response should expose the underlying InferenceResult so
    downstream code can inspect anchors, activations, subgraph."""
    r = respond(backbone, "What is photosynthesis?")
    assert r.inference is not None
    assert "photosynthesis" in {t.surface for t in r.inference.tokens}
    assert r.primary_anchor_id in r.inference.anchors.values()
    assert len(r.inference.subgraph_node_ids) > 0


def test_question_answer_capitalized_and_punctuated(backbone):
    """Answers should be sentence-cased and end with a period."""
    r = respond(backbone, "What is photosynthesis?")
    assert r.text[0].isupper(), r.text
    assert r.text.rstrip().endswith("."), r.text
