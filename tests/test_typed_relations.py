from strands.codebook import Codebook
from strands.codon import Codon
from strands.comparator import compare_strands
from strands.relations import RelationType, TypedRelation
from strands.strand import CodonEntry, Strand


def _entry(codon: Codon, *rels: TypedRelation) -> CodonEntry:
    return CodonEntry(codon=codon, shade=0x55, related=tuple(rels))


def test_synonym_relation_boosts_cross_codon_match():
    a = Codon(0x03, 0, 1)
    b = Codon(0x03, 0, 2)

    plain = compare_strands(Strand([_entry(a)]), Strand([_entry(b)])).score
    typed = compare_strands(
        Strand([_entry(a, TypedRelation(b, 255, RelationType.SYNONYM))]),
        Strand([_entry(b)]),
    ).score

    assert typed > plain
    assert typed > 0.85


def test_antonym_relation_penalizes_same_domain_match():
    a = Codon(0x03, 0, 1)
    b = Codon(0x03, 0, 2)

    plain = compare_strands(Strand([_entry(a)]), Strand([_entry(b)])).score
    opposed = compare_strands(
        Strand([_entry(a, TypedRelation(b, 255, RelationType.ANTONYM))]),
        Strand([_entry(b)]),
    ).score

    assert plain > 0.25
    assert opposed < plain
    assert opposed <= 0.31


def test_codebook_accepts_typed_relation_entries():
    cb = Codebook({
        "entries": {
            "cold": {"d": "QU", "c": 1, "n": 2, "trel": [["ANTI", "QU103", 240]]},
        }
    })

    entry = cb.lookup("cold")

    assert entry is not None
    assert entry.related[0].relation == RelationType.ANTONYM
    assert entry.related[0].codon == Codon(0x03, 1, 3)
    assert entry.related[0].weight == 240


def test_code_search_profile_uses_query_coverage():
    q = Strand([
        CodonEntry(Codon(0x0C, 3, 1), 0x55, word="file"),
        CodonEntry(Codon(0x01, 5, 9), 0x55, word="open"),
    ])
    target = Strand([
        CodonEntry(Codon(0x0C, 3, 1), 0x55, word="file"),
        CodonEntry(Codon(0x01, 5, 9), 0x55, word="open"),
        CodonEntry(Codon(0x04, 1, 2), 0x55, word="helper"),
        CodonEntry(Codon(0x04, 1, 3), 0x55, word="buffer"),
    ])

    default = compare_strands(q, target, profile="default").score
    code_search = compare_strands(q, target, profile="code_search").score

    assert code_search > default
    assert code_search > 0.9
