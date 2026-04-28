"""Shared utilities for the benchmark test suite."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_pairs_tsv(path: Path, *, delimiter: str = "\t", min_cols: int = 3) -> list[tuple[str, str, float]]:
    """Load (word_a, word_b, gold) triples from a TSV/CSV file."""
    pairs: list[tuple[str, str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(delimiter)
        if len(parts) < min_cols:
            continue
        try:
            pairs.append((parts[0].strip(), parts[1].strip(), float(parts[2])))
        except ValueError:
            continue
    return pairs


def load_men3000(path: Path | None = None) -> list[tuple[str, str, float]]:
    p = path or DATA_DIR / "men3000.txt"
    return load_pairs_tsv(p, delimiter="\t")


def load_rg65(path: Path | None = None) -> list[tuple[str, str, float]]:
    p = path or DATA_DIR / "rg65.txt"
    return load_pairs_tsv(p, delimiter="\t")


def load_simverb(path: Path | None = None) -> list[tuple[str, str, float]]:
    p = path or DATA_DIR / "simverb3500.txt"
    return load_pairs_tsv(p, delimiter="\t")


def load_sts(path: Path) -> list[tuple[str, str, float]]:
    """Load STS-style file: <gold>\\t<sentence_a>\\t<sentence_b>."""
    pairs: list[tuple[str, str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            pairs.append((parts[1].strip(), parts[2].strip(), float(parts[0])))
        except (ValueError, IndexError):
            continue
    return pairs


def load_sick(path: Path) -> list[tuple[str, str, float]]:
    """Load SICK file: pair_ID\\tsentence_A\\tsentence_B\\trelatedness\\tentailment."""
    pairs: list[tuple[str, str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:  # skip header
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            pairs.append((parts[1].strip(), parts[2].strip(), float(parts[3])))
        except (ValueError, IndexError):
            continue
    return pairs
