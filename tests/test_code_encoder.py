from strands import compare_strands, encode_code, encode

PY_FETCH = """
async def fetch_user(user_id):
    response = await fetch(f'/api/users/{user_id}')
    if not response.ok:
        raise Error('Not found')
    return response.json()
"""

JS_FETCH = """
async function fetchUser(id) {
  const response = await fetch('/api/users/' + id);
  if (!response.ok) throw new Error('Not found');
  return response.json();
}
"""

SQL_QUERY = "SELECT name, age FROM users WHERE active = 1"


def test_python_encodes():
    r = encode_code(PY_FETCH, language="python")
    assert r.byte_size > 0
    assert r.structural_count > 0
    assert r.semantic_count > 0


def test_javascript_encodes():
    r = encode_code(JS_FETCH, language="javascript")
    assert r.byte_size > 0
    assert r.structural_count > 0


def test_cross_language_similarity():
    """Equivalent Python and JS code should be more similar than Python ↔ SQL."""
    py = encode_code(PY_FETCH, language="python").strand
    js = encode_code(JS_FETCH, language="javascript").strand
    sql = encode_code(SQL_QUERY, language="python").strand

    py_js = compare_strands(py, js).score
    py_sql = compare_strands(py, sql).score

    assert py_js > py_sql, f"Cross-lang similarity {py_js:.3f} should beat SQL {py_sql:.3f}"
    assert py_js >= 0.5, f"Equivalent Python/JS expected ≥ 0.5, got {py_js:.3f}"


def test_keyword_routing():
    """Code keywords route to code domains, not text."""
    r = encode_code("if x: return x", language="python")
    domains = {e.codon.domain_code for e in r.strand.codons}
    # Should include CF (control flow) for if/return
    from strands.codon import DOMAIN_CODES

    cf_id = DOMAIN_CODES["CF"]
    assert any(e.codon.domain == cf_id for e in r.strand.codons), (
        f"Expected CF domain in {[e.codon.to_str() for e in r.strand.codons]}"
    )


def test_string_literals_use_text_domains():
    """String contents should be encoded via the text pipeline."""
    r = encode_code('print("happy dog runs")', language="python")
    # "happy" should appear in EM (Emotion) domain via text encoding
    from strands.codon import DOMAIN_CODES

    em_id = DOMAIN_CODES["EM"]
    assert any(e.codon.domain == em_id for e in r.strand.codons)


def test_density_metric():
    src_keyword_heavy = "if x: while y: try: return"
    src_identifier_heavy = "user response context request"
    kr = encode_code(src_keyword_heavy, language="python")
    ir = encode_code(src_identifier_heavy, language="python")
    assert kr.structural_density > ir.structural_density


def test_unknown_language_falls_back():
    """An unknown language still encodes — keywords just won't be recognized."""
    r = encode_code("def foo(): return 1", language="unknown_lang")
    assert r.byte_size >= 0  # may have only identifier-derived codons
