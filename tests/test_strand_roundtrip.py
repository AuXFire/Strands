import random

from strands.codon import Codon
from strands.strand import CodonEntry, Strand


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
    for a, b in zip(s.codons, s2.codons):
        assert a.codon == b.codon
        assert a.shade == b.shade


def test_byte_size():
    s = _random_strand(5)
    raw = s.to_binary()
    assert len(raw) == s.byte_size
    # v2 default: 8-byte header + 12 bytes per token
    # v2 default: 8-byte header + 4 + 4*4 = 20 bytes per token
    assert s.byte_size == 8 + 20 * 5
    # v1 mode is also supported
    raw_v1 = s.to_binary(version=1)
    assert len(raw_v1) == 8 + 4 * 5
