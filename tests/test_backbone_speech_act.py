"""Speech-act tagging tests (BDRM §3.2 enhancement).

Verifies the SpeechAct descriptor that augments intent + question_type
with negation, hedging, sentiment, and a top-level act category.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    DiscourseState,
    StubComputeModule,
    classify_speech_act,
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


def test_question_act():
    sa = classify_speech_act(
        "What is a cat?", intent="question_answering",
        question_type="definition",
    )
    assert sa.act == "question"
    assert sa.subtype == "definition"


def test_yesno_subtype():
    sa = classify_speech_act(
        "Is a cat a dog?", intent="question_answering",
        question_type="yesno",
    )
    assert sa.subtype == "yesno"


def test_assertion_with_negation():
    sa = classify_speech_act(
        "Cats are not cute.", intent="inform", question_type="",
    )
    assert sa.act == "assertion"
    assert sa.has_negation is True
    assert sa.polarity == -1


def test_assertion_with_hedge():
    sa = classify_speech_act(
        "I think cats are cute.", intent="inform", question_type="",
    )
    assert sa.hedged is True
    # Hedged but not negated.
    assert sa.has_negation is False


def test_directive():
    sa = classify_speech_act(
        "Tell me more.", intent="instruction", question_type="",
    )
    assert sa.act == "directive"


def test_expressive_greeting():
    sa = classify_speech_act(
        "Hello there", intent="social", question_type="",
    )
    assert sa.act == "expressive"
    assert sa.subtype == "greeting"


def test_expressive_thanks():
    sa = classify_speech_act(
        "Thanks for your help.", intent="social", question_type="",
    )
    assert sa.subtype == "thanks"
    assert sa.sentiment == 1


def test_expressive_apology():
    sa = classify_speech_act(
        "Sorry to bother you.", intent="social", question_type="",
    )
    assert sa.subtype == "apology"


def test_speech_act_lands_in_conditioning(backbone):
    """The SpeechActTag should appear on Conditioning.speech_act
    when the Compute Module fires."""
    cm = StubComputeModule(override="ok")
    state = DiscourseState()
    respond(backbone, "What is xyzzy?", state=state, compute=cm)
    assert len(cm.calls) == 1
    sa = cm.calls[0].speech_act
    assert sa is not None
    assert sa.act == "question"


def test_polarity_default_positive():
    sa = classify_speech_act(
        "Cats are cute.", intent="inform", question_type="",
    )
    assert sa.polarity == 1


def test_negation_in_question_flips_polarity():
    sa = classify_speech_act(
        "Isn't a cat an animal?", intent="question_answering",
        question_type="yesno",
    )
    # 'isn't' contains the n't cue.
    assert sa.has_negation is True
    assert sa.polarity == -1
