"""Optional tree-sitter AST encoder tests. Skip when tree-sitter unavailable."""

import pytest

from strands import compare_strands

ts = pytest.importorskip("tree_sitter")
from strands.code_ast import encode_ast, is_available  # noqa: E402


PY_FUNC = "def add(a, b):\n    return a + b\n"
PY_CLASS = "class Foo:\n    def bar(self):\n        return 1\n"


@pytest.mark.skipif(not is_available("python"), reason="tree-sitter-python not installed")
def test_ast_encodes_python():
    r = encode_ast(PY_FUNC, language="python")
    assert r.byte_size > 0
    assert r.structural_count > 0
    assert r.semantic_count > 0


@pytest.mark.skipif(not is_available("python"), reason="tree-sitter-python not installed")
def test_ast_recognizes_function():
    r = encode_ast(PY_FUNC, language="python")
    # function_definition -> "function" -> MD
    from strands.codon import DOMAIN_CODES

    md_id = DOMAIN_CODES["MD"]
    assert any(e.codon.domain == md_id for e in r.strand.codons)


@pytest.mark.skipif(not is_available("python"), reason="tree-sitter-python not installed")
def test_ast_class_vs_function():
    """Class and function should produce different but related strands."""
    f = encode_ast(PY_FUNC, language="python")
    c = encode_ast(PY_CLASS, language="python")
    score = compare_strands(f.strand, c.strand).score
    assert 0.0 < score <= 1.0
