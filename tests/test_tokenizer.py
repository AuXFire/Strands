from strands.tokenizer import tokenize


def test_tokenize_drops_punctuation():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_tokenize_drops_stop_words():
    tokens = tokenize("The quick brown fox jumps over the lazy dog")
    assert "the" not in tokens
    assert "over" not in tokens
    assert "quick" in tokens
    assert "fox" in tokens


def test_tokenize_keeps_stop_words_when_disabled():
    tokens = tokenize("the dog", drop_stop_words=False)
    assert tokens == ["the", "dog"]


def test_tokenize_handles_apostrophes():
    tokens = tokenize("don't worry")
    assert "worry" in tokens
