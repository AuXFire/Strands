"""Multi-turn discourse + pronoun resolution tests (BDRM §3.3).

Verifies that a follow-up question containing a pronoun ("it", "they",
"this") resolves against the rolling DiscourseState topic and answers
about the previously-mentioned entity.

The headline test is the two-turn "What is a cat?" → "What is it?"
flow: turn 2 must produce the same definition as turn 1 because "it"
resolves to the cat anchor from turn 1's discourse state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    DiscourseState,
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


def test_two_turn_pronoun_resolution(backbone):
    """Headline multi-turn: 'What is a cat?' → 'What is it?' should
    answer about cat both times. The second turn carries no anchor of
    its own — resolution must come from the discourse state."""
    state = DiscourseState()
    r1 = respond(backbone, "What is a cat?", state=state)
    r2 = respond(backbone, "What is it?", state=state)

    assert r1.primary_anchor_id is not None
    assert r1.pronoun_resolved is False
    # Turn 2 has no direct anchor — the only path to a real answer is
    # through pronoun resolution.
    assert r2.pronoun_resolved is True
    assert r2.primary_anchor_id == r1.primary_anchor_id
    # Same anchor → same gloss-backed answer.
    assert r2.text == r1.text


def test_pronoun_without_prior_topic(backbone):
    """A pronoun in the very first turn has nothing to resolve against
    — the response should fall through to the no-anchor fallback and
    flag for the Compute Module."""
    state = DiscourseState()
    r = respond(backbone, "What is it?", state=state)
    assert r.pronoun_resolved is False
    assert r.primary_anchor_id is None
    assert r.needs_compute_module is True


def test_pronoun_resolution_discounts_confidence(backbone):
    """Pronoun-resolved answers are heuristic-bound — confidence should
    be lower than the equivalent direct-anchor answer."""
    state = DiscourseState()
    r_direct = respond(backbone, "What is photosynthesis?", state=DiscourseState())
    respond(backbone, "What is photosynthesis?", state=state)
    r_pronoun = respond(backbone, "What is it?", state=state)
    assert r_pronoun.pronoun_resolved is True
    assert r_pronoun.confidence < r_direct.confidence


def test_low_content_turn_preserves_topic(backbone):
    """A non-entity-anchored turn ('Tell me more.') must not clobber
    the rolling topic — pronoun resolution after it should still work.
    """
    state = DiscourseState()
    respond(backbone, "What is photosynthesis?", state=state)
    head_after_turn_1 = state.active_topic_node_ids[0]

    respond(backbone, "Tell me more.", state=state)
    head_after_turn_2 = state.active_topic_node_ids[0]
    # Topic must survive the verb-anchored intermediate turn.
    assert head_after_turn_2 == head_after_turn_1

    r = respond(backbone, "What is it?", state=state)
    assert r.pronoun_resolved is True
    assert r.primary_anchor_id == head_after_turn_1
    assert "photosynthesis" in r.text.lower()


def test_topic_swaps_on_new_entity(backbone):
    """A new ENTITY-type anchor in a later turn should overtake the
    rolling topic — pronouns after it resolve to the new entity."""
    state = DiscourseState()
    respond(backbone, "What is a cat?", state=state)
    cat_anchor = state.active_topic_node_ids[0]

    respond(backbone, "What is a dog?", state=state)
    dog_anchor = state.active_topic_node_ids[0]
    assert dog_anchor != cat_anchor

    r = respond(backbone, "What is it?", state=state)
    assert r.pronoun_resolved is True
    # 'it' should now resolve to dog, not cat.
    assert r.primary_anchor_id == dog_anchor


def test_they_and_this_also_resolve(backbone):
    """Pronoun set covers 'they', 'this', 'that', not just 'it'."""
    for pronoun in ("they", "this", "that"):
        state = DiscourseState()
        respond(backbone, "What is a cat?", state=state)
        r = respond(backbone, f"What is {pronoun}?", state=state)
        assert r.pronoun_resolved is True, (
            f"pronoun {pronoun!r} did not resolve"
        )


def test_entity_register_tracks_surfaces(backbone):
    """The entity_register should accumulate surface→node_id pairs
    across turns for future direct-reference lookup."""
    state = DiscourseState()
    respond(backbone, "What is a cat?", state=state)
    respond(backbone, "What is photosynthesis?", state=state)
    assert "cat" in state.entity_register
    assert "photosynthesis" in state.entity_register
    assert state.entity_register["cat"] != state.entity_register["photosynthesis"]


def test_turn_count_increments_regardless(backbone):
    """Every respond() call must advance turn_count, even on no-anchor
    or social turns."""
    state = DiscourseState()
    respond(backbone, "What is a cat?", state=state)
    respond(backbone, "Hello", state=state)
    respond(backbone, "Tell me more.", state=state)
    respond(backbone, "What is it?", state=state)
    assert state.turn_count == 4
