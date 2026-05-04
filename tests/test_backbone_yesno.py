"""Yes/no question tests (BDRM §3.4 enhancement).

Verifies that polar questions (Is X a Y? Can X Y? Does X have Y?)
produce binary verdicts grounded in backbone evidence: hypernym
closure for category checks, direct edge presence for capability
and meronymy. Negation is explicit ("No, X is not a Y; X is a Z.")
so the NN can later be trained to mirror the structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands.backbone import (
    answer_yesno,
    is_yesno_question,
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


def test_is_yesno_question_predicate():
    for p in (
        "Is a cat an animal?", "Are dogs mammals?", "Can fish swim?",
        "Does a car have wheels?", "Will it rain?", "Should we go?",
    ):
        assert is_yesno_question(p), p
    for p in (
        "What is a cat?", "Where do birds live?", "Tell me about cats.",
        "Hello", "I love programming",
    ):
        assert not is_yesno_question(p), p


def test_hypernym_yes_via_canonical_chain(backbone):
    r = respond(backbone, "Is a cat an animal?")
    assert r.text.lower().startswith("yes"), r.text
    assert r.confidence >= 0.9


def test_hypernym_yes_for_direct_parent(backbone):
    r = respond(backbone, "Is a cat a feline?")
    assert r.text.lower().startswith("yes"), r.text


def test_hypernym_yes_for_distant_ancestor(backbone):
    r = respond(backbone, "Is a cat a mammal?")
    assert r.text.lower().startswith("yes"), r.text


def test_hypernym_no_with_actual_category(backbone):
    """A NO verdict should explain what the subject actually IS."""
    r = respond(backbone, "Is a cat a dog?")
    assert r.text.lower().startswith("no"), r.text
    assert "is a" in r.text.lower()  # the corrective phrase


def test_dog_not_a_cat(backbone):
    r = respond(backbone, "Is a dog a cat?")
    assert r.text.lower().startswith("no"), r.text


def test_capable_of_yes(backbone):
    """Cross-sense edge match: 'Can a bird fly?' must find the
    bird→fly CAPABLE_OF edge even though both lemmas have many senses."""
    r = respond(backbone, "Can a bird fly?")
    assert r.text.lower().startswith("yes"), r.text
    assert "bird can fly" in r.text.lower()


def test_capable_of_yes_for_fish(backbone):
    r = respond(backbone, "Can a fish swim?")
    assert r.text.lower().startswith("yes"), r.text


def test_articles_an_for_vowels(backbone):
    """'an animal' not 'a animal' — vowel article rule applies."""
    r = respond(backbone, "Is a cat an animal?")
    assert "is an animal" in r.text.lower(), r.text


def test_articles_a_for_consonants(backbone):
    r = respond(backbone, "Is a cat a feline?")
    assert "is a feline" in r.text.lower(), r.text


def test_marginal_subject_sense_does_not_cause_false_yes(backbone):
    """'cat' has a caterpillar/construction-equipment sense that's
    technically a vehicle. We restrict to top-3 subject senses by
    edge count so the dominant animal sense governs the verdict."""
    r = respond(backbone, "Is a cat a vehicle?")
    assert r.text.lower().startswith("no"), r.text


def test_unsure_when_no_relation_evidence(backbone):
    """Absence of evidence ≠ evidence of absence. When no edge supports
    a positive answer for a capability/property/has question, the
    verdict is 'unsure', not 'no'."""
    r = respond(backbone, "Can a fish fly?")
    assert "evidence" in r.text.lower() or "certain" in r.text.lower(), r.text
    assert r.confidence < 0.5  # flags Compute Module


def test_yesno_does_not_clobber_other_intents(backbone):
    """A non-yes/no question still goes through the normal Q&A path."""
    r = respond(backbone, "What is a cat?")
    # Definition path produces gloss, not yes/no.
    assert not r.text.lower().startswith("yes")
    assert not r.text.lower().startswith("no")


def test_answer_yesno_returns_structured_verdict(backbone):
    """Direct call to answer_yesno should expose the verdict shape
    so downstream code can route on it."""
    yn = answer_yesno(backbone, "Is a cat an animal?")
    assert yn is not None
    assert yn.verdict == "yes"
    assert yn.confidence >= 0.9
    yn = answer_yesno(backbone, "Is a cat a dog?")
    assert yn is not None
    assert yn.verdict == "no"


def test_answer_yesno_returns_none_for_non_yesno(backbone):
    """Called on a wh-question, answer_yesno should return None
    so the caller can route to the normal Q&A path."""
    assert answer_yesno(backbone, "What is a cat?") is None


def test_does_x_have_y_pattern(backbone):
    """The 'Does X have Y?' pattern must take priority over the
    generic 'Does X Y?' pattern — otherwise 'Does a car have wheels?'
    matches with target='have wheels'."""
    yn = answer_yesno(backbone, "Does a car have wheels?")
    assert yn is not None
    # Either yes or unsure depending on the data — but never with
    # 'have wheels' as the target (would indicate the generic
    # pattern beat the specific one).
    assert yn.target_surface == "wheels"
