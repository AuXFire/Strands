"""Document-level summarization tests."""

from strands.document import (
    DocumentFingerprint,
    fingerprint_similarity,
    histogram_cosine,
    topcodon_jaccard,
)


def test_fingerprint_extracts_top_codons():
    text = "happy joyful cheerful glad happy happy"
    fp = DocumentFingerprint.from_text(text)
    codons = [c for c, _ in fp.top_codons]
    # All emotion codons (EM domain) should dominate
    assert all(c.startswith("EM") for c in codons[:3])


def test_fingerprint_storage_smaller_than_strand():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A determined cat watched from the window. "
        "Birds were singing in the morning sunshine. "
        "Children played happily in the garden. "
    ) * 10
    fp = DocumentFingerprint.from_text(text)
    assert fp.fingerprint_bytes < fp.byte_size
    # Compression ratio should be meaningful for long docs
    assert fp.fingerprint_bytes < fp.byte_size * 0.5


def test_similar_documents_score_high():
    a = DocumentFingerprint.from_text(
        "The dog chased the cat through the garden. "
        "Birds fluttered overhead, and the sun shone brightly."
    )
    b = DocumentFingerprint.from_text(
        "A cat ran from a dog in the yard. "
        "Birds flew above as the sunshine warmed everything."
    )
    c = DocumentFingerprint.from_text(
        "The function compiles the source code into bytecode. "
        "Errors are reported with line numbers and stack traces."
    )

    sim_ab = fingerprint_similarity(a, b)
    sim_ac = fingerprint_similarity(a, c)

    assert sim_ab > sim_ac, f"sim_ab {sim_ab:.3f} should beat sim_ac {sim_ac:.3f}"


def test_histogram_cosine_basics():
    assert histogram_cosine({"EM": 1}, {"EM": 1}) == 1.0
    assert histogram_cosine({"EM": 1}, {"AC": 1}) == 0.0
    assert 0.0 < histogram_cosine({"EM": 2, "AC": 1}, {"EM": 1, "AC": 1}) < 1.0


def test_topcodon_jaccard_basics():
    a = [("EM000", 5), ("AC001", 3)]
    b = [("EM000", 7), ("NA000", 2)]
    j = topcodon_jaccard(a, b)
    assert j == 1 / 3  # one shared (EM000), three total
