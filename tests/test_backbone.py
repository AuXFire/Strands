"""Backbone builder + loader smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strands.backbone import EDGE_DTYPE, NODE_DTYPE, ConceptType, Rel, load
from strands.backbone import builder as backbone_builder


def test_node_dtype_is_128_bytes():
    assert NODE_DTYPE.itemsize == 128


def test_edge_dtype_is_32_bytes():
    assert EDGE_DTYPE.itemsize == 32


def test_relation_taxonomy_has_stable_ids():
    # Pin a few canonical IDs so external storage stays compatible.
    assert int(Rel.HYPERNYM) == 0x0001
    assert int(Rel.HYPONYM) == 0x0002
    assert int(Rel.SYNONYM) == 0x0005
    assert int(Rel.ANTONYM) == 0x0006
    assert int(Rel.RELATED_TO) == 0x00B0


@pytest.fixture(scope="module")
def small_backbone(tmp_path_factory):
    """A tiny WordNet-only backbone for fast tests."""
    out_dir = tmp_path_factory.mktemp("backbone")
    backbone_builder.build(
        out_dir,
        conceptnet_path=None,
        max_synsets=2000,
    )
    return load(out_dir)


def test_load_returns_correct_counts(small_backbone):
    assert small_backbone.node_count == 2000
    assert small_backbone.edge_count >= 0


def test_node_lookup_returns_lemmas(small_backbone):
    # First node has a node_id of 0 and a non-empty lemma list.
    node = small_backbone.node(0)
    assert node.node_id == 0
    assert isinstance(node.lemmas, tuple)


def test_edges_are_in_relation_taxonomy(small_backbone):
    # Pick a node with at least one edge.
    edge_counts = small_backbone.nodes["relationship_count"]
    nz = np.where(edge_counts > 0)[0]
    if nz.size == 0:
        pytest.skip("no edges in the small slice")
    node_id = int(nz[0])
    edges = small_backbone.edges_of(node_id)
    valid_rels = {int(r) for r in Rel}
    for e in edges:
        assert int(e["relation_type"]) in valid_rels


def test_codon_link_set_for_first_lemma(small_backbone):
    # Most synsets' first lemma is in the codebook → codon should be set.
    n_with_codon = int((small_backbone.nodes["codon_domain"] != 0xFF).sum())
    assert n_with_codon > small_backbone.node_count * 0.5, (
        f"only {n_with_codon}/{small_backbone.node_count} nodes have codon links"
    )


def test_edge_offsets_are_valid(small_backbone):
    # Every edge offset + count should fall within the edge array.
    n = small_backbone.node_count
    e = small_backbone.edge_count
    for i in range(n):
        node = small_backbone.nodes[i]
        offset = int(node["relationship_offset"])
        count = int(node["relationship_count"])
        assert offset + count <= e, f"node {i} edges out of range"


def test_lemma_index_finds_known_word(small_backbone):
    # Grab some real lemma from a node and verify reverse lookup finds it.
    for i in range(min(50, small_backbone.node_count)):
        lemmas = small_backbone.lemmas_for(i)
        if lemmas:
            target_lemma = lemmas[0]
            ids = small_backbone.nodes_for_lemma(target_lemma)
            assert i in ids, f"lemma '{target_lemma}' didn't find node {i}"
            return
    pytest.fail("no node had any lemmas in first 50")
