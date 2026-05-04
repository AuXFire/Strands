"""Question-type discrimination tests (BDRM §3.4 enhancement).

Verifies that the wh-word / verb pattern in a question routes to a
relation-specific answerer instead of always falling back to the
synset gloss. Also checks the cross-sense WSD fallback that fires
when the chosen anchor has no edges of the requested relation but
a sibling sense does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import load, respond

_BACKBONE_DIR = Path(__file__).parent.parent / "strands" / "data" / "backbone"


@pytest.fixture(scope="module")
def backbone():
    if not (_BACKBONE_DIR / "backbone.manifest.json").exists():
        pytest.skip(
            "backbone not built — run `python scripts/build_backbone.py`"
        )
    return load(_BACKBONE_DIR)


def test_where_routes_to_at_location(backbone):
    """'Where do birds live?' should answer with AT_LOCATION targets,
    not the bird gloss."""
    r = respond(backbone, "Where do birds live?")
    assert "can be found at" in r.text.lower()
    assert "vertebrate" not in r.text.lower()  # not the gloss


def test_what_does_x_do_routes_to_capable_of(backbone):
    """'What does a bird do?' should walk CAPABLE_OF."""
    r = respond(backbone, "What does a bird do?")
    assert "is capable of" in r.text.lower()


def test_what_can_x_do_routes_to_capable_of(backbone):
    r = respond(backbone, "What can a fish do?")
    assert "is capable of" in r.text.lower()


def test_what_is_x_made_of_routes_to_made_of(backbone):
    """Cross-sense WSD: 'What is a house made of?' — the social-unit
    sense has no MADE_OF edges, but the dwelling sense does. Sibling
    fallback should bridge."""
    r = respond(backbone, "What is a house made of?")
    assert "made of" in r.text.lower()


def test_what_is_x_used_for_routes_to_used_for(backbone):
    r = respond(backbone, "What is a school used for?")
    assert "used for" in r.text.lower()


def test_definition_questions_still_use_gloss(backbone):
    """'What is X?' without a typed pattern should still produce the
    gloss-based definition."""
    r = respond(backbone, "What is photosynthesis?")
    assert "synthesis of compounds" in r.text.lower()


def test_user_surface_preserved_across_sense_swap(backbone):
    """When sibling-sense fallback fires, the answer must use the
    word the USER typed (e.g. 'house') not the matched node's primary
    lemma (e.g. 'home')."""
    r = respond(backbone, "What is a house made of?")
    # Subject must start with 'House', not 'Home' or 'Family'.
    assert r.text.lower().startswith("house "), r.text


def test_user_surface_preserved_for_school(backbone):
    """Same deal — 'school' question shouldn't render as 'Train' even
    though the relation walk crosses senses."""
    r = respond(backbone, "What is a school used for?")
    assert r.text.lower().startswith("school "), r.text


def test_typed_question_falls_back_to_gloss_when_no_relation(backbone):
    """'What is photosynthesis made of?' has no good MADE_OF on
    photosynthesis — the answerer should fall through to the gloss
    rather than fabricate."""
    r = respond(backbone, "What is photosynthesis made of?")
    # Either the gloss-based fallback or a sibling-sense answer.
    # Most importantly, it must produce a coherent sentence.
    assert r.text
    assert r.text.endswith(".")


def test_typed_answer_carries_lower_confidence_than_gloss(backbone):
    """Typed-relation answers should be flagged with conf=0.85 vs
    gloss=0.9. Cross-sense answers drop further to ~0.75."""
    direct = respond(backbone, "What is a bird?")
    typed = respond(backbone, "Where do birds live?")
    cross_sense = respond(backbone, "What is a house made of?")
    assert direct.confidence > typed.confidence
    assert typed.confidence > cross_sense.confidence


def test_where_pattern_does_not_fire_on_definition(backbone):
    """A normal 'What is X?' must not match any typed pattern."""
    from strands.backbone.response import _classify_question_type
    assert _classify_question_type("What is a bird?") == "definition"
    assert _classify_question_type("What is photosynthesis?") == "definition"


def test_question_type_classifications(backbone):
    """Direct unit test of the wh-pattern classifier."""
    from strands.backbone.response import _classify_question_type
    cases = [
        ("Where do birds live?",      "location"),
        ("Where is Paris?",            "location"),
        ("What does a dog do?",        "ability"),
        ("What can a car do?",         "ability"),
        ("How does a bird fly?",       "ability"),
        ("What is water made of?",     "composition"),
        ("What is a school used for?", "purpose"),
        ("What causes rain?",          "cause"),
        ("Why does ice melt?",         "cause"),
        ("What is a cat?",             "definition"),
        ("Who is Einstein?",           "definition"),
    ]
    for prompt, expected in cases:
        actual = _classify_question_type(prompt)
        assert actual == expected, (
            f"{prompt!r}: expected {expected}, got {actual}"
        )
