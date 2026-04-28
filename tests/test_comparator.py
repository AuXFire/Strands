from strands import compare


def test_synonym_score_high():
    assert compare("happy", "joyful").score > 0.8


def test_unrelated_score_zero():
    assert compare("happy", "database").score < 0.15


def test_same_domain_different_concept():
    score = compare("happy", "sad").score
    assert 0.15 <= score <= 0.5


def test_self_compare():
    assert compare("happy dog", "happy dog").score > 0.95
