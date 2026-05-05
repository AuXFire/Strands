"""Tests for the realization layer (BDRM §3.5 / B1).

Cover the data structures, store lookup, slot filling rules, and
the slot/compute interchange (compute-pending fallback when a
non-COMPUTE slot has no filler).
"""

from __future__ import annotations

import pytest

from strands.backbone.schema import Rel
from strands.realization import (
    FillerValue,
    Register,
    SEED_TEMPLATES,
    Segment,
    SlotKind,
    SlotSpec,
    Template,
    TemplateStore,
    build_default_store,
    render,
    render_text,
)


# --- Data structures ---------------------------------------------------


def test_template_construction():
    t = Template(
        template_id="t1", pattern="{x} is {y}.",
        slots=(
            SlotSpec(name="x", kind=SlotKind.LEMMA),
            SlotSpec(name="y", kind=SlotKind.LEMMA),
        ),
        response_shape="example",
    )
    assert t.template_id == "t1"
    assert len(t.slots) == 2
    assert t.weight == 1.0  # default


def test_segment_defaults():
    s = Segment(text="hi", source="literal")
    assert s.confidence == 1.0
    assert s.backbone_node_id == -1
    assert s.pending_slot is None


# --- Store -------------------------------------------------------------


def test_store_add_and_lookup_by_shape():
    store = TemplateStore()
    t = Template(
        template_id="x.1", pattern="{a}.",
        slots=(SlotSpec(name="a", kind=SlotKind.LEMMA),),
        response_shape="x",
    )
    store.add(t)
    assert len(store) == 1
    assert store.get("x.1") is t
    assert store.by_shape("x") == [t]
    assert store.by_shape("y") == []


def test_store_ranks_by_weight():
    store = TemplateStore()
    low = Template(
        template_id="low", pattern="low",
        response_shape="x", weight=0.5,
    )
    high = Template(
        template_id="high", pattern="high",
        response_shape="x", weight=1.5,
    )
    store.add(low)
    store.add(high)
    assert store.by_shape("x")[0] is high
    assert store.best(shape="x") is high


def test_store_filters_by_register():
    store = TemplateStore()
    formal = Template(
        template_id="formal", pattern="…",
        response_shape="x", register=Register.FORMAL,
    )
    informal = Template(
        template_id="informal", pattern="…",
        response_shape="x", register=Register.INFORMAL,
    )
    store.add(formal)
    store.add(informal)
    assert store.by_shape("x", register=Register.FORMAL) == [formal]
    assert store.by_shape("x", register=Register.INFORMAL) == [informal]


def test_store_lookup_by_relation():
    store = TemplateStore()
    t = Template(
        template_id="rel.hyp",
        pattern="{x} is a kind of {y}.",
        slots=(
            SlotSpec(name="x", kind=SlotKind.LEMMA),
            SlotSpec(name="y", kind=SlotKind.LEMMA),
        ),
        response_shape="elaborate",
        relation_type=int(Rel.HYPERNYM),
    )
    store.add(t)
    assert store.by_relation(int(Rel.HYPERNYM)) == [t]
    assert store.by_relation(int(Rel.MERONYM)) == []


def test_store_rejects_duplicate_id():
    store = TemplateStore()
    t = Template(template_id="dup", pattern="x")
    store.add(t)
    with pytest.raises(ValueError, match="duplicate"):
        store.add(t)


# --- Default store (seeds) --------------------------------------------


def test_default_store_loads_all_seeds():
    store = build_default_store()
    assert len(store) == len(SEED_TEMPLATES)


def test_default_store_has_template_per_response_shape():
    """Every shape the planner can request should resolve to at
    least one template."""
    expected = {
        "definition", "definition_with_hypernym", "definition_fallback",
        "definition_unknown",
        "yesno_yes_hypernym", "yesno_no_with_actual", "yesno_no_plain",
        "yesno_yes_capable", "yesno_yes_property", "yesno_yes_has",
        "yesno_unsure",
        "answer_location", "answer_composition", "answer_purpose",
        "answer_ability", "answer_cause", "answer_effect",
        "elaborate", "elaborate_exhausted",
        "inform_belief_ack", "inform_generic_ack", "inform_bare",
        "instruct_with_subject", "instruct_bare",
        "social_greeting", "social_thanks", "social_apology",
        "social_farewell", "social_fallback",
        "fallback_no_anchor", "fallback_rephrase",
    }
    store = build_default_store()
    for shape in expected:
        assert store.best(shape=shape) is not None, f"missing: {shape}"


def test_default_store_relation_coverage():
    """Every relation the elaboration walker uses should have a
    template tagged with that relation_type."""
    store = build_default_store()
    walked = (
        Rel.HYPERNYM, Rel.HAS_PROPERTY, Rel.MERONYM, Rel.HOLONYM,
        Rel.AT_LOCATION, Rel.USED_FOR, Rel.CAPABLE_OF, Rel.MADE_OF,
        Rel.HYPONYM,
    )
    for rel in walked:
        assert store.best(relation_type=int(rel)) is not None, (
            f"no template for relation {rel.name}"
        )


# --- Rendering: deterministic paths -----------------------------------


def test_render_definition_basic():
    store = build_default_store()
    t = store.best(shape="definition")
    out = render(t, {
        "subject": FillerValue("Cat"),
        "gloss": FillerValue("any of several large cats"),
    })
    assert out.finalize() == "Cat is any of several large cats."
    assert not out.has_pending_compute


def test_render_yesno_yes_with_articles():
    store = build_default_store()
    t = store.best(shape="yesno_yes_hypernym")
    out = render(t, {
        "subject": FillerValue("cat"),
        "target": FillerValue("animal"),
    })
    # Article rule: 'a cat', 'an animal'
    assert "a cat" in out.finalize()
    assert "an animal" in out.finalize()


def test_render_yesno_no_three_slots():
    store = build_default_store()
    t = store.best(shape="yesno_no_with_actual")
    out = render(t, {
        "subject": FillerValue("cat"),
        "target": FillerValue("dog"),
        "actual": FillerValue("felid"),
    })
    assert out.finalize() == "No, a cat is not a dog; a cat is a felid."


def test_render_lemma_list_two_items():
    store = build_default_store()
    t = store.best(shape="answer_location")
    out = render(t, {
        "subject": FillerValue("Bird"),
        "target": ["sky", "tree"],
    })
    assert out.finalize() == "Bird can be found at sky and tree."


def test_render_lemma_list_three_items_oxford():
    store = build_default_store()
    t = store.best(shape="answer_location")
    out = render(t, {
        "subject": FillerValue("Bird"),
        "target": ["sky", "tree", "bush"],
    })
    assert out.finalize() == "Bird can be found at sky, tree, and bush."


def test_render_lemma_list_single_item():
    store = build_default_store()
    t = store.best(shape="answer_ability")
    out = render(t, {
        "subject": FillerValue("Fish"),
        "target": ["swim"],
    })
    assert out.finalize() == "Fish is capable of swim."


def test_render_inform_ack_uses_literal():
    store = build_default_store()
    t = store.best(shape="inform_belief_ack")
    out = render(t, {"raw": FillerValue("Cats are cute", source="literal")})
    assert out.finalize() == "Got it — noted that Cats are cute."


def test_render_relation_lookup():
    store = build_default_store()
    t = store.best(relation_type=int(Rel.HYPERNYM))
    out = render(t, {
        "subject": FillerValue("Cat"),
        "target": FillerValue("feline"),
    })
    assert out.finalize() == "Cat is a kind of feline."


# --- Slot/compute interchange -----------------------------------------


def test_missing_slot_becomes_compute_pending():
    """A non-COMPUTE slot without a filler degrades to a
    compute_pending segment so the realization completes."""
    store = build_default_store()
    t = store.best(shape="definition")
    out = render(t, {"subject": FillerValue("Mystery")})  # no gloss
    assert out.has_pending_compute
    pending = out.pending_slots
    assert len(pending) == 1
    assert pending[0].pending_slot.name == "gloss"
    # The placeholder text marks the gap.
    assert "⟨gloss⟩" in out.finalize()


def test_explicit_compute_slot():
    """A SlotKind.COMPUTE slot is always compute-pending, even if a
    filler is supplied — the planner explicitly delegated it."""
    t = Template(
        template_id="creative",
        pattern="{lead} {opener}",
        slots=(
            SlotSpec(name="lead", kind=SlotKind.LITERAL),
            SlotSpec(
                name="opener", kind=SlotKind.COMPUTE,
                compute_intervention="creative",
            ),
        ),
        response_shape="creative_open",
    )
    out = render(t, {
        "lead": FillerValue("Once upon a time,", source="literal"),
        "opener": FillerValue("ignored", source="literal"),
    })
    pending = out.pending_slots
    assert len(pending) == 1
    assert pending[0].pending_intervention == "creative"


def test_disabled_compute_fallback_emits_empty():
    """A template with allow_compute_fallback=False emits an empty
    segment (confidence 0) for missing slots so the caller knows
    rendering failed."""
    t = Template(
        template_id="strict",
        pattern="{x} is {y}.",
        slots=(
            SlotSpec(name="x", kind=SlotKind.LEMMA),
            SlotSpec(name="y", kind=SlotKind.LEMMA),
        ),
        response_shape="strict",
        allow_compute_fallback=False,
    )
    out = render(t, {"x": FillerValue("Cat")})
    # The 'y' slot becomes an empty literal with conf 0.
    assert not out.has_pending_compute
    assert any(
        s.source == "literal" and s.text == "" and s.confidence == 0.0
        for s in out.segments
    )


def test_unknown_placeholder_emitted_literally():
    """A placeholder with no matching SlotSpec becomes a literal
    segment with conf 0 — surfaces the planner bug instead of
    silently dropping it."""
    t = Template(
        template_id="bug",
        pattern="{x} {missing}",
        slots=(SlotSpec(name="x", kind=SlotKind.LEMMA),),
        response_shape="bug",
    )
    out = render(t, {"x": FillerValue("hi")})
    text = out.finalize()
    assert "{missing}" in text


def test_render_overall_confidence_min_of_segments():
    t = Template(
        template_id="conf",
        pattern="{x} is {y}.",
        slots=(
            SlotSpec(name="x", kind=SlotKind.LEMMA),
            SlotSpec(name="y", kind=SlotKind.LEMMA),
        ),
        response_shape="conf",
    )
    out = render(t, {
        "x": FillerValue("a", confidence=0.9),
        "y": FillerValue("b", confidence=0.6),
    })
    assert out.confidence == pytest.approx(0.6)


def test_render_text_convenience():
    t = Template(
        template_id="conv",
        pattern="{x}.",
        slots=(SlotSpec(name="x", kind=SlotKind.LEMMA),),
        response_shape="conv",
    )
    assert render_text(t, {"x": FillerValue("hello")}) == "hello."


# --- Inflection edge cases --------------------------------------------


def test_article_an_for_initial_vowel():
    t = Template(
        template_id="art",
        pattern="{x}",
        slots=(SlotSpec(
            name="x", kind=SlotKind.NOUN_PHRASE, article=True,
        ),),
        response_shape="art",
    )
    assert render_text(t, {"x": FillerValue("apple")}) == "an apple"
    assert render_text(t, {"x": FillerValue("orange")}) == "an orange"


def test_article_a_for_initial_consonant():
    t = Template(
        template_id="art2",
        pattern="{x}",
        slots=(SlotSpec(
            name="x", kind=SlotKind.NOUN_PHRASE, article=True,
        ),),
        response_shape="art2",
    )
    assert render_text(t, {"x": FillerValue("dog")}) == "a dog"


def test_capitalize_first_letter():
    t = Template(
        template_id="cap",
        pattern="{x}.",
        slots=(SlotSpec(
            name="x", kind=SlotKind.LEMMA, capitalize=True,
        ),),
        response_shape="cap",
    )
    assert render_text(t, {"x": FillerValue("cat")}) == "Cat."


def test_gloss_lowercased():
    """Gloss segments lowercase the first character so the surrounding
    sentence reads naturally."""
    t = Template(
        template_id="g",
        pattern="X is {g}.",
        slots=(SlotSpec(name="g", kind=SlotKind.GLOSS),),
        response_shape="g",
    )
    out = render_text(t, {"g": FillerValue("Any of several cats")})
    assert out == "X is any of several cats."
