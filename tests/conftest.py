"""Shared pytest fixtures: ensure NLTK corpora are available."""

from __future__ import annotations

import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_nltk_data():
    for corpus in ("wordnet", "sentiwordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)
    for tagger in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(f"taggers/{tagger}")
        except LookupError:
            nltk.download(tagger, quiet=True)
