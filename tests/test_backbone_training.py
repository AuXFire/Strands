"""Training plumbing tests (BDRM §7 prep).

Verifies that Conditioning round-trips cleanly through JSON, and
that the RecordingComputeModule captures every turn into a JSONL
file ready for offline NN training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands.backbone import (
    DiscourseState,
    RecordingComputeModule,
    StubComputeModule,
    TrainingExample,
    conditioning_to_dict,
    dict_to_conditioning,
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


def test_conditioning_round_trip(backbone):
    """A Conditioning serialized to dict and back should preserve
    all the fields the NN needs."""
    cm = StubComputeModule(override="ok")
    state = DiscourseState()
    respond(backbone, "Cats are cute", state=state)
    respond(backbone, "What is xyzzy?", state=state, compute=cm)
    original = cm.calls[0]

    d = conditioning_to_dict(original)
    # Survives a full JSON round-trip.
    j = json.dumps(d)
    d2 = json.loads(j)
    rebuilt = dict_to_conditioning(d2)

    assert rebuilt.prompt == original.prompt
    assert rebuilt.intent == original.intent
    assert rebuilt.deterministic_answer == original.deterministic_answer
    assert rebuilt.deterministic_confidence == pytest.approx(
        original.deterministic_confidence,
    )
    assert len(rebuilt.history) == len(original.history)
    assert len(rebuilt.user_beliefs) == len(original.user_beliefs)
    if original.user_beliefs:
        assert (
            rebuilt.user_beliefs[0].subject_lemma
            == original.user_beliefs[0].subject_lemma
        )
    if original.speech_act is not None:
        assert rebuilt.speech_act is not None
        assert rebuilt.speech_act.act == original.speech_act.act


def test_recording_module_captures_each_turn(backbone, tmp_path):
    """Every prompt that triggers the Compute Module path should
    produce one captured example."""
    out = tmp_path / "training.jsonl"
    state = DiscourseState()
    with RecordingComputeModule(out) as rec:
        respond(backbone, "What is xyzzy?", state=state, compute=rec)
        respond(backbone, "What is plugh?", state=state, compute=rec)
    assert rec.count == 2
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    # Every line is valid JSON and has the required keys.
    for line in lines:
        d = json.loads(line)
        assert "conditioning" in d
        assert "deterministic_answer" in d
        assert "deterministic_confidence" in d
        assert d["conditioning"]["prompt"]


def test_recording_module_preserves_history(backbone, tmp_path):
    """A multi-turn session should accumulate history in each
    captured example."""
    out = tmp_path / "training.jsonl"
    state = DiscourseState()
    with RecordingComputeModule(out) as rec:
        # Force every turn through the seam with a high floor.
        for p in [
            "What is a cat?",
            "Tell me more.",
            "What is it?",
            "Is a cat an animal?",
        ]:
            respond(
                backbone, p, state=state, compute=rec,
                confidence_floor=0.99,
            )
    assert rec.count == 4
    examples = [
        TrainingExample(**{k: v for k, v in json.loads(line).items() if k != "deterministic_confidence"} | {"deterministic_confidence": json.loads(line)["deterministic_confidence"]})
        for line in out.read_text().splitlines()
    ]
    # History grows monotonically.
    history_lengths = [len(e.conditioning["history"]) for e in examples]
    assert history_lengths == sorted(history_lengths)
    assert history_lengths[0] == 0
    assert history_lengths[-1] == 3  # three prior turns by turn 4


def test_recording_module_inner_pass_through(backbone, tmp_path):
    """When wrapped around an inner compute module, the recorder
    should pass through the inner's override."""
    out = tmp_path / "training.jsonl"
    inner = StubComputeModule(override="[inner override]")
    rec = RecordingComputeModule(out, inner=inner)
    try:
        state = DiscourseState()
        r = respond(backbone, "What is xyzzy?", state=state, compute=rec)
    finally:
        rec.close()
    assert r.text == "[inner override]"
    # And the recorder captured the override alongside the conditioning.
    line = out.read_text().splitlines()[0]
    d = json.loads(line)
    assert d["override"] == "[inner override]"


def test_recording_module_preserves_user_beliefs(backbone, tmp_path):
    """Beliefs accumulated across turns should appear in later
    captured examples."""
    out = tmp_path / "training.jsonl"
    state = DiscourseState()
    with RecordingComputeModule(out) as rec:
        respond(backbone, "Cats are cute", state=state)
        respond(backbone, "Dogs can swim", state=state)
        respond(
            backbone, "What is xyzzy?", state=state, compute=rec,
        )
    line = out.read_text().splitlines()[0]
    d = json.loads(line)
    beliefs = d["conditioning"]["user_beliefs"]
    assert len(beliefs) == 2
    subjects = [b["subject"] for b in beliefs]
    assert "cats" in subjects
    assert "dogs" in subjects


def test_conditioning_dict_is_json_serializable(backbone):
    """conditioning_to_dict() output must be json.dumps-safe — no
    numpy scalars, no Path objects, no enums."""
    cm = StubComputeModule(override="ok")
    state = DiscourseState()
    respond(backbone, "What is xyzzy?", state=state, compute=cm)
    d = conditioning_to_dict(cm.calls[0])
    # Must not raise.
    json.dumps(d)


def test_recording_module_appends_when_reopened(backbone, tmp_path):
    """The recorder opens the file in append mode so a fresh
    instance over the same path adds to existing lines."""
    out = tmp_path / "training.jsonl"
    state = DiscourseState()
    rec1 = RecordingComputeModule(out)
    respond(backbone, "What is xyzzy?", state=state, compute=rec1)
    rec1.close()
    rec2 = RecordingComputeModule(out)
    respond(backbone, "What is plugh?", state=state, compute=rec2)
    rec2.close()
    assert len(out.read_text().splitlines()) == 2
