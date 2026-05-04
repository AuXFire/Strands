"""Topic-elaboration tests (BDRM §3.4 enhancement).

Verifies that 'Tell me more' / 'continue' / 'what else' style prompts
walk the backbone's relation edges from the active topic and produce
novel facts on each successive call. When the relation budget is
exhausted, confidence drops so the Compute Module path activates.
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


def test_tell_me_more_emits_distinct_facts(backbone):
    """Three successive 'tell me more' calls after a question should
    produce three different sentences — each backed by a distinct
    relation edge."""
    state = DiscourseState()
    respond(backbone, "What is photosynthesis?", state=state)
    facts = []
    for _ in range(4):
        r = respond(backbone, "Tell me more.", state=state)
        facts.append(r.text)
    # All four must differ — the elaboration walker tracks emitted
    # facts in DiscourseState.emitted_facts.
    assert len(set(facts)) == len(facts), facts


def test_elaboration_phrasings_all_route(backbone):
    """'tell me more', 'continue', 'and', 'what else', 'go on' all
    route to the topic walker."""
    for phrase in ("Tell me more.", "Continue", "And?", "What else?", "Go on"):
        state = DiscourseState()
        respond(backbone, "What is photosynthesis?", state=state)
        # First elaboration in a fresh state should hit the walker
        # and produce a sentence about photosynthesis.
        r = respond(backbone, phrase, state=state)
        assert "photosynthesis" in r.text.lower(), (
            f"phrase {phrase!r} did not elaborate, got {r.text!r}"
        )


def test_elaboration_exhaustion_lowers_confidence(backbone):
    """After all reachable facts have been emitted, the walker should
    return a low-confidence fallback so the Compute Module kicks in."""
    state = DiscourseState()
    respond(backbone, "What is a dog?", state=state)
    # Fire many elaborations until exhaustion is reached.
    last = None
    for _ in range(50):
        last = respond(backbone, "Tell me more.", state=state)
        if "all I have" in last.text:
            break
    assert last is not None
    assert "all I have" in last.text
    assert last.confidence < 0.5
    assert last.needs_compute_module is True


def test_topic_swap_yields_independent_facts(backbone):
    """Switching the topic should produce a new stream of facts —
    emitted_facts is keyed by (subject, rel, target) so cat-facts
    don't suppress dog-facts."""
    state = DiscourseState()
    respond(backbone, "What is a cat?", state=state)
    cat_fact1 = respond(backbone, "Tell me more.", state=state).text
    respond(backbone, "What is a dog?", state=state)
    dog_fact1 = respond(backbone, "Tell me more.", state=state).text
    assert cat_fact1 != dog_fact1
    # And cat-facts continue to produce novel content if we go back.
    respond(backbone, "What is a cat?", state=state)
    cat_fact2 = respond(backbone, "Tell me more.", state=state).text
    assert cat_fact2 != cat_fact1


def test_elaboration_without_active_topic_falls_through(backbone):
    """'Tell me more' with an empty discourse state should fall
    through to the regular instruction template, not crash."""
    state = DiscourseState()
    r = respond(backbone, "Tell me more.", state=state)
    # No topic to elaborate on → instruction template fires.
    assert r.inference.intent == "instruction"
    assert "all I have" not in r.text


def test_elaboration_records_facts_in_state(backbone):
    """Each successful elaboration appends to state.emitted_facts."""
    state = DiscourseState()
    respond(backbone, "What is photosynthesis?", state=state)
    assert len(state.emitted_facts) == 0
    respond(backbone, "Tell me more.", state=state)
    assert len(state.emitted_facts) == 1
    respond(backbone, "Tell me more.", state=state)
    assert len(state.emitted_facts) == 2


def test_elaboration_anchor_is_active_topic(backbone):
    """The Response.primary_anchor_id should be the topic node — so
    downstream consumers (e.g. Compute Module conditioning) can see
    what subject the response is about."""
    state = DiscourseState()
    respond(backbone, "What is a cat?", state=state)
    cat_node = state.active_topic_node_ids[0]
    r = respond(backbone, "Tell me more.", state=state)
    assert r.primary_anchor_id == cat_node


def test_pronoun_then_elaborate(backbone):
    """Multi-turn coreference + elaboration: 'What is X?' →
    'What is it?' → 'Tell me more.' should all stay anchored on X."""
    state = DiscourseState()
    respond(backbone, "What is photosynthesis?", state=state)
    r2 = respond(backbone, "What is it?", state=state)
    r3 = respond(backbone, "Tell me more.", state=state)
    assert r2.pronoun_resolved is True
    assert "photosynthesis" in r3.text.lower()
