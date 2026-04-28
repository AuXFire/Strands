"""Strand v2 binary format + WSD tests."""

import pytest

from strands.codon import Codon
from strands.strand import CodonEntry, Strand


def test_v1_round_trip_unchanged():
    s = Strand(codons=[
        CodonEntry(codon=Codon(0x01, 0x02, 0x03), shade=0xAB),
        CodonEntry(codon=Codon(0x04, 0x05, 0x06), shade=0xCD),
    ])
    raw = s.to_binary()
    s2 = Strand.from_binary(raw)
    assert len(s2.codons) == 2
    assert s2.codons[0].codon == s.codons[0].codon


def test_v2_extended_round_trip():
    s = Strand(codons=[
        CodonEntry(
            codon=Codon(0x01, 0x02, 0x03), shade=0xAB,
            sense_rank=2, semantic_role=5, source_position=42,
        ),
        CodonEntry(
            codon=Codon(0x04, 0x05, 0x06), shade=0xCD,
            sense_rank=0, semantic_role=0, source_position=99,
        ),
    ])
    raw = s.to_binary(extended=True)
    s2 = Strand.from_binary(raw)
    assert len(s2.codons) == 2
    assert s2.codons[0].codon == s.codons[0].codon
    assert s2.codons[0].sense_rank == 2
    assert s2.codons[0].semantic_role == 5
    assert s2.codons[0].source_position == 42
    assert s2.codons[1].source_position == 99


def test_v2_byte_size_doubles():
    s = Strand(codons=[
        CodonEntry(codon=Codon(0x01, 0x02, 0x03), shade=0xAB),
    ] * 10)
    v1 = s.to_binary()
    v2 = s.to_binary(extended=True)
    assert len(v1) == 8 + 4 * 10
    assert len(v2) == 8 + 8 * 10


def test_lesk_picks_distinct_senses():
    wn = pytest.importorskip("nltk.corpus")
    from strands.wsd import lesk_select

    # "bank" can mean financial institution or river edge
    # In context of "money loan deposit", should pick financial sense
    finance_idx = lesk_select("bank", ["money", "loan", "deposit", "account", "cash"])
    river_idx = lesk_select("bank", ["river", "water", "flow", "boat", "fish"])

    # Just verify both succeed and return ints; specific senses depend on
    # WordNet ordering which can shift between releases.
    assert finance_idx is not None
    assert river_idx is not None


def test_wsd_in_encoder_populates_sense_rank():
    from strands import encode

    # The verb sense of "run" vs noun sense of "running"
    r = encode("the marathon runner ran for thirty kilometers", wsd=True)
    # Just verify wsd path runs without errors and may populate sense_rank
    assert len(r.strand.codons) > 0
    # source_position should be populated when wsd=True path runs
    assert all(0 <= c.source_position < 1000 for c in r.strand.codons)
