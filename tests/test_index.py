from strands.index import InMemoryIndex


def test_search_ranks_synonym_first():
    index = InMemoryIndex()
    index.add("a", "happy joyful cheerful")
    index.add("b", "rocket spacecraft launch")
    index.add("c", "database server query")

    results = index.search("joyful")
    assert len(results) >= 1
    assert results[0].entry.id == "a"


def test_domain_pre_filter():
    index = InMemoryIndex()
    index.add("emotion", "love hate")
    index.add("tech", "computer database")

    results = index.search("happy", top_k=5)
    ids = [r.entry.id for r in results]
    assert "emotion" in ids
