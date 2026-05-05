"""Tests for the response structure tree (BDRM §3.4.3, B2)."""

from __future__ import annotations

import pytest

from strands.realization import (
    ActType,
    CommunicativeAct,
    ConceptualLeaf,
    FillerValue,
    RealizedResponse,
    ResponseStructure,
    build_default_store,
    realize,
    sequence,
    single_leaf,
)


# --- Data types ---------------------------------------------------------


def test_conceptual_leaf_defaults():
    leaf = ConceptualLeaf()
    assert leaf.template_id == ""
    assert leaf.template_shape == ""
    assert leaf.template_relation == 0
    assert leaf.confidence == 1.0
    assert leaf.flagged_for_compute is False
    assert leaf.fillers == {}


def test_communicative_act_leaf_only():
    leaf = ConceptualLeaf(template_shape="x")
    act = CommunicativeAct(act_type=ActType.ASSERT, leaf=leaf)
    assert act.is_leaf
    assert list(act.walk_leaves()) == [leaf]


def test_communicative_act_children_walk_in_order():
    leaves = [ConceptualLeaf(template_shape=f"l{i}") for i in range(3)]
    act = CommunicativeAct(
        act_type=ActType.SEQUENCE,
        children=[
            CommunicativeAct(act_type=ActType.ASSERT, leaf=l)
            for l in leaves
        ],
    )
    assert not act.is_leaf
    assert list(act.walk_leaves()) == leaves


def test_response_structure_walk_leaves():
    s = single_leaf(
        ActType.ASSERT, template_shape="t",
        fillers={"x": FillerValue("y")},
    )
    leaves = list(s.walk_leaves())
    assert len(leaves) == 1
    assert leaves[0].template_shape == "t"


def test_has_pending_compute_flagged_leaf():
    s = single_leaf(
        ActType.ASSERT, template_shape="t",
        flagged_for_compute=True, intervention="creative",
    )
    assert s.has_pending_compute


def test_has_pending_compute_no_flag():
    s = single_leaf(
        ActType.ASSERT, template_shape="definition",
        fillers={"subject": FillerValue("a"), "gloss": FillerValue("b")},
    )
    assert not s.has_pending_compute


# --- Constructors -------------------------------------------------------


def test_single_leaf_builds_one_leaf_tree():
    s = single_leaf(
        ActType.ACKNOWLEDGE,
        template_shape="inform_bare",
        confidence=0.6,
    )
    assert s.root.is_leaf
    assert s.root.act_type == ActType.ACKNOWLEDGE
    assert s.root.confidence == 0.6
    assert s.confidence == 0.6


def test_sequence_min_confidence():
    a = CommunicativeAct(
        act_type=ActType.ASSERT, leaf=ConceptualLeaf(),
        confidence=0.9,
    )
    b = CommunicativeAct(
        act_type=ActType.ASSERT, leaf=ConceptualLeaf(),
        confidence=0.7,
    )
    seq = sequence([a, b])
    assert seq.act_type == ActType.SEQUENCE
    assert seq.confidence == pytest.approx(0.7)
    assert len(seq.children) == 2


# --- Realization --------------------------------------------------------


@pytest.fixture(scope="module")
def store():
    return build_default_store()


def test_realize_single_leaf_definition(store):
    s = single_leaf(
        ActType.ASSERT,
        template_shape="definition",
        fillers={
            "subject": FillerValue("Cat"),
            "gloss": FillerValue("any of several large cats"),
        },
        confidence=0.9,
    )
    out = realize(s, store)
    assert isinstance(out, RealizedResponse)
    assert out.text == "Cat is any of several large cats."
    assert out.confidence == pytest.approx(0.9)
    assert not out.has_pending_compute


def test_realize_sequence_joins_with_separator(store):
    s = ResponseStructure(root=sequence([
        CommunicativeAct(
            act_type=ActType.ASSERT,
            leaf=ConceptualLeaf(
                template_shape="definition",
                fillers={
                    "subject": FillerValue("Bird"),
                    "gloss": FillerValue("warm-blooded vertebrate"),
                },
            ),
        ),
        CommunicativeAct(
            act_type=ActType.ELABORATE,
            leaf=ConceptualLeaf(
                template_shape="elaborate",
                template_relation=0x0001,  # HYPERNYM
                fillers={
                    "subject": FillerValue("Bird"),
                    "target": FillerValue("vertebrate"),
                },
            ),
        ),
    ]))
    out = realize(s, store)
    assert "Bird is warm-blooded vertebrate." in out.text
    assert "Bird is a kind of vertebrate." in out.text
    # Default separator is a single space.
    assert ". B" in out.text


def test_realize_compute_flagged_leaf_pending(store):
    s = single_leaf(
        ActType.ASSERT,
        template_shape="definition",
        flagged_for_compute=True,
        intervention="creative",
    )
    out = realize(s, store)
    assert out.has_pending_compute
    assert len(out.pending_segments) == 1
    assert out.pending_segments[0].pending_intervention == "creative"


def test_realize_missing_template_pending(store):
    s = single_leaf(
        ActType.ASSERT,
        template_shape="this_shape_does_not_exist",
    )
    out = realize(s, store)
    assert out.has_pending_compute


def test_realize_missing_filler_per_slot_pending(store):
    """A non-COMPUTE slot whose filler is missing becomes a per-slot
    compute_pending segment inside the rendered template."""
    s = single_leaf(
        ActType.ASSERT,
        template_shape="definition",
        fillers={"subject": FillerValue("Mystery")},  # no gloss
    )
    out = realize(s, store)
    assert out.has_pending_compute
    pending = out.pending_segments
    # Renderer emits one segment for the missing 'gloss' slot.
    assert any(
        s.pending_slot is not None and s.pending_slot.name == "gloss"
        for s in pending
    )


def test_realize_template_id_overrides_shape(store):
    """When both template_id and template_shape are set, id wins."""
    s = single_leaf(
        ActType.ASSERT,
        template_id="def.basic",      # specific
        template_shape="elaborate",   # would otherwise win lookup
        fillers={
            "subject": FillerValue("X"),
            "gloss": FillerValue("y"),
        },
    )
    out = realize(s, store)
    assert out.text == "X is y."


def test_realize_relation_only_lookup(store):
    """A leaf with relation_type but no shape resolves via the
    relation-only index."""
    s = single_leaf(
        ActType.ELABORATE,
        template_relation=0x0001,  # HYPERNYM
        fillers={
            "subject": FillerValue("Cat"),
            "target": FillerValue("feline"),
        },
    )
    out = realize(s, store)
    # Highest-weight HYPERNYM template is elab.hypernym ("X is a kind of Y.").
    assert out.text == "Cat is a kind of feline."
