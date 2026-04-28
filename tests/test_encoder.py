from strands.encoder import encode


def test_encode_basic():
    result = encode("happy dog runs fast")
    assert len(result.strand.codons) >= 3
    assert result.byte_size == 8 + 4 * len(result.strand.codons)


def test_encode_lemmatizes_inflected():
    r = encode("dogs running")
    words = [c.word for c in r.strand.codons]
    # lemmatized forms should appear, not raw inflections
    assert "dog" in words
    assert "run" in words or any(c.word == "run" for c in r.strand.codons)


def test_encode_records_unknowns():
    r = encode("xyzzy plugh")
    assert "xyzzy" in r.unknowns
    assert "plugh" in r.unknowns


def test_encode_deterministic():
    a = encode("the happy dog")
    b = encode("the happy dog")
    assert a.strand_text == b.strand_text
