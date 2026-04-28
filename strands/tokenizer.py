"""Tokenization + stop-word filtering."""

from __future__ import annotations

import re

# Minimal English stop-word list (~150 words per spec §7.1).
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "and", "or", "but", "nor", "so", "yet",
    "in", "on", "at", "by", "for", "with", "from", "to", "of", "as",
    "into", "onto", "upon", "within", "without",
    "if", "then", "else", "than",
    "when", "where", "why", "how", "what", "who", "whom", "which",
    "would", "could", "should", "may", "might", "must", "shall", "will", "can",
    "not", "no", "yes",
    "there", "here",
    "out", "off", "over", "under", "up", "down",
    "all", "any", "some", "each", "every", "both", "either", "neither",
    "such", "same", "other", "another",
    "very", "too", "more", "most", "less", "least",
})

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z]")


def tokenize(text: str, *, drop_stop_words: bool = True) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if drop_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens
