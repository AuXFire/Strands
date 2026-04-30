from strands.encoder import encode


def test_encode_basic():
    result = encode("happy dog runs fast")
    assert len(result.strand.codons) >= 3
    # strand v3 default: 8-byte header + 32 bytes per token
    assert result.byte_size == 8 + 32 * len(result.strand.codons)


def test_encode_handles_inflected_forms():
    """Inflected forms encode to the same codon as their base lemma."""
    inflected = encode("dogs running")
    base = encode("dog run")
    assert len(inflected.strand.codons) == len(base.strand.codons)
    for a, b in zip(inflected.strand.codons, base.strand.codons):
        assert a.codon == b.codon


def test_encode_records_unknowns():
    r = encode("xyzzy plugh")
    assert "xyzzy" in r.unknowns
    assert "plugh" in r.unknowns


def test_encode_deterministic():
    a = encode("the happy dog")
    b = encode("the happy dog")
    assert a.strand_text == b.strand_text
