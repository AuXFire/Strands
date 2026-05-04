"""Inform-turn belief extraction tests (BDRM §3.3 / §2.3 volatility).

Verifies that declarative user statements ('Cats are cute.', 'Dogs
can swim.') get lifted into structured Belief records, accumulate on
the DiscourseState, and surface in the Compute Module conditioning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    Belief,
    BeliefRecord,
    DiscourseState,
    Rel,
    StubComputeModule,
    extract_belief,
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


def test_extract_property_belief(backbone):
    b = extract_belief(backbone, "Cats are cute")
    assert b is not None
    assert b.subject_surface == "cats"
    assert b.target_surface == "cute"
    assert b.relation == Rel.HAS_PROPERTY


def test_extract_hypernym_belief(backbone):
    b = extract_belief(backbone, "Paris is a city")
    assert b is not None
    assert b.subject_surface == "paris"
    assert b.target_surface == "city"
    assert b.relation == Rel.HYPERNYM


def test_extract_capable_belief(backbone):
    b = extract_belief(backbone, "Dogs can swim")
    assert b is not None
    assert b.subject_surface == "dogs"
    assert b.target_surface == "swim"
    assert b.relation == Rel.CAPABLE_OF


def test_extract_has_a_belief(backbone):
    b = extract_belief(backbone, "Cars have wheels")
    assert b is not None
    assert b.relation == Rel.HAS_A


def test_extract_at_location_belief(backbone):
    b = extract_belief(backbone, "Birds live in trees")
    assert b is not None
    assert b.relation == Rel.AT_LOCATION


def test_extract_causes_belief(backbone):
    b = extract_belief(backbone, "Sugar causes cavities")
    assert b is not None
    assert b.relation == Rel.CAUSES


def test_pronoun_subject_skipped(backbone):
    """First-person and pronominal subjects don't extract — would
    require coreference + speaker model we don't have yet."""
    for p in ("I love programming", "It is raining", "They are happy"):
        assert extract_belief(backbone, p) is None, p


def test_questions_dont_extract(backbone):
    """Question forms must not match the declarative patterns."""
    for p in ("Are cats cute?", "What is a cat?", "Where do birds live?"):
        assert extract_belief(backbone, p) is None, p


def test_belief_resolves_node_ids(backbone):
    """When subject and target are known to the backbone, the Belief
    should carry their node IDs for later use."""
    b = extract_belief(backbone, "Cats are cute")
    assert b is not None
    assert b.subject_node_id != -1
    assert b.target_node_id != -1


def test_unresolved_target_keeps_surface(backbone):
    """If a surface can't be resolved, the Belief keeps the surface
    string and node_id=-1 — the Compute Module can still see the
    assertion."""
    b = extract_belief(backbone, "Frobozz are wibblefrots")
    if b is not None:
        # Either skipped or kept with -1 IDs; both are acceptable.
        assert b.subject_node_id == -1 or b.target_node_id == -1


def test_inform_turn_records_belief_in_state(backbone):
    """A respond() call on a declarative should append to
    state.session_beliefs."""
    state = DiscourseState()
    respond(backbone, "Cats are cute", state=state)
    assert len(state.session_beliefs) == 1
    b = state.session_beliefs[0]
    assert b.subject_surface == "cats"
    assert b.relation == Rel.HAS_PROPERTY


def test_inform_acknowledgement_echoes_user_wording(backbone):
    """The ack should use the user's exact phrasing — avoids subject-
    verb-agreement bugs ('cats is cute') and confirms the registered
    fact verbatim."""
    state = DiscourseState()
    r = respond(backbone, "Cats are cute", state=state)
    assert "cats are cute" in r.text.lower()
    assert "noted that" in r.text.lower()
    assert r.confidence >= 0.8


def test_inform_acknowledgement_falls_back_when_no_extraction(backbone):
    """A declarative we can't structure ('I love programming') still
    produces a clean acknowledgement, just the generic one."""
    state = DiscourseState()
    r = respond(backbone, "I love programming", state=state)
    assert "got it" in r.text.lower()
    assert len(state.session_beliefs) == 0


def test_beliefs_accumulate_across_turns(backbone):
    """Multiple inform turns should stack on session_beliefs in order."""
    state = DiscourseState()
    respond(backbone, "Cats are cute", state=state)
    respond(backbone, "Dogs can swim", state=state)
    respond(backbone, "Paris is a city", state=state)
    assert len(state.session_beliefs) == 3
    assert state.session_beliefs[0].relation == Rel.HAS_PROPERTY
    assert state.session_beliefs[1].relation == Rel.CAPABLE_OF
    assert state.session_beliefs[2].relation == Rel.HYPERNYM


def test_beliefs_carried_in_conditioning(backbone):
    """The Compute Module conditioning should expose user_beliefs
    so the NN can see what the user has told us."""
    cm = StubComputeModule(override="ok")
    state = DiscourseState()
    respond(backbone, "Cats are cute", state=state)
    respond(backbone, "Dogs can swim", state=state)
    # Force a low-confidence prompt so the CM fires.
    respond(backbone, "What is xyzzy?", state=state, compute=cm)
    assert len(cm.calls) == 1
    c = cm.calls[0]
    assert len(c.user_beliefs) == 2
    assert isinstance(c.user_beliefs[0], BeliefRecord)
    assert c.user_beliefs[0].subject_lemma == "cats"
    assert c.user_beliefs[0].relation == "HAS_PROPERTY"
    assert c.user_beliefs[1].subject_lemma == "dogs"
    assert c.user_beliefs[1].relation == "CAPABLE_OF"


def test_belief_turn_index_set(backbone):
    """Each belief should know which turn it was stated in."""
    state = DiscourseState()
    respond(backbone, "Cats are cute", state=state)  # turn 1
    respond(backbone, "Dogs can swim", state=state)  # turn 2
    assert state.session_beliefs[0].turn_index == 1
    assert state.session_beliefs[1].turn_index == 2
