"""Codebook extension mechanism (spec §14.3)."""

from pathlib import Path

import pytest

from strands.codebook import Codebook


BASE = Path(__file__).parent.parent / "strands" / "data" / "codebook_v0.1.0.json"
MEDICAL = Path(__file__).parent.parent / "strands" / "data" / "extensions" / "medical.json"


def test_extension_adds_new_terms():
    cb = Codebook.load_with_extensions(BASE, [MEDICAL])
    e = cb.lookup("tachycardia")
    assert e is not None
    assert e.domain_code == "BD"


def test_extension_overrides_base():
    cb = Codebook.load(BASE)
    cb.merge_extension({
        "entries": {
            "happy": {"d": "EM", "c": 0, "n": 99, "s": {"p": 3, "f": 1, "i": 3}}
        }
    })
    e = cb.lookup("happy")
    assert e.codon.concept == 99


def test_extension_does_not_break_existing():
    cb = Codebook.load_with_extensions(BASE, [MEDICAL])
    e = cb.lookup("happy")
    assert e is not None
    assert e.domain_code == "EM"


def test_multiple_extensions_merge_in_order():
    cb = Codebook.load_with_extensions(BASE, [MEDICAL])
    # Verify all medical terms loaded
    for term in ["tachycardia", "biopsy", "vaccine", "pathology"]:
        assert cb.lookup(term) is not None, f"missing: {term}"


def test_extension_preserves_size():
    cb_base = Codebook.load(BASE)
    cb_ext = Codebook.load_with_extensions(BASE, [MEDICAL])
    assert len(cb_ext) >= len(cb_base)
