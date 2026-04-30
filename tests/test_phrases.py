from strands import compare, encode, encode_code


def _words(text: str) -> set[str]:
    return {entry.word for entry in encode(text).strand.codons}


def test_text_encoder_adds_open_file_frame():
    assert "open_file" in _words("how to open a text file on python")


def test_code_encoder_adds_open_file_frame_from_identifiers():
    strand = encode_code("def read_file(path):\n    return open(path).read()").strand
    assert "open_file" in {entry.word for entry in strand.codons}


def test_phrase_frame_boosts_related_intent():
    framed = compare("open a text file", "read file source").score
    unrelated = compare("open a text file", "sort numeric list").score
    assert framed > unrelated
