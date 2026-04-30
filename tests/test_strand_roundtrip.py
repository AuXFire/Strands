import random

from strands.codon import Codon
from strands.relations import RelationType, TypedRelation
from strands.strand import VERSION_V3, CodonEntry, Strand


def _random_strand(n: int) -> Strand:
    rng = random.Random(42)
    return Strand(
        codons=[
            CodonEntry(
                codon=Codon(
                    domain=rng.randint(0, 0x12),
                    category=rng.randint(0, 15),
                    concept=rng.randint(0, 255),
                ),
                shade=rng.randint(0, 255),
            )
            for _ in range(n)
        ]
    )


def test_text_round_trip():
    s = _random_strand(8)
    assert Strand.from_text(s.to_text()).to_text() == s.to_text()


def test_binary_round_trip():
    s = _random_strand(20)
    raw = s.to_binary()
    s2 = Strand.from_binary(raw)
    assert len(s2.codons) == len(s.codons)
    assert s2.version == VERSION_V3
    for a, b in zip(s.codons, s2.codons):
        assert a.codon == b.codon
        assert a.shade == b.shade


def test_byte_size():
    s = _random_strand(5)
    raw = s.to_binary()
    assert len(raw) == s.byte_size
    # v3 default: 8-byte header + 8-byte primary frame + 4 typed relation slots.
    assert s.byte_size == 8 + 32 * 5
    # v1 mode is also supported
    raw_v1 = s.to_binary(version=1)
    assert len(raw_v1) == 8 + 4 * 5


def test_v3_typed_relations_round_trip():
    target = Codon(domain=0x03, category=2, concept=7)
    s = Strand(codons=[
        CodonEntry(
            codon=Codon(domain=0x03, category=2, concept=6),
            shade=0x55,
            role=3,
            features=0x1234,
            sense=2,
            related=(TypedRelation(target, 240, RelationType.ANTONYM),),
        )
    ])

    s2 = Strand.from_binary(s.to_binary())
    entry = s2.codons[0]
    assert entry.role == 3
    assert entry.features == 0x1234
    assert entry.sense == 2
    assert entry.related[0].codon == target
    assert entry.related[0].relation == RelationType.ANTONYM
    assert entry.related[0].weight == 240
