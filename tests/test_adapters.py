from strands.adapters import get_scoring_profile, iter_frame_specs, load_adapters
from strands.codon import Codon
from strands.relations import RelationType


def test_builtin_adapters_load():
    adapters = load_adapters()
    assert {"strict", "topical", "sentence", "code_search", "phrase_frames"} <= set(adapters)


def test_code_search_adapter_profile_overrides_weights():
    profile = get_scoring_profile("code_search")
    assert profile.query_coverage is True
    assert profile.lexical_weight == 0.52
    assert profile.relation_scale[RelationType.USED_FOR] == 0.82


def test_topical_adapter_profile_is_association_heavy():
    profile = get_scoring_profile("topical")
    assert profile.antonym_penalty == 0.08
    assert profile.relation_scale[RelationType.RELATED] == 1.10
    assert profile.relation_scale[RelationType.ASSOCIATED] == 0.80
    assert profile.relation_scale[RelationType.CONTEXT] == 0.90


def test_phrase_frame_adapter_loads_frame_codons():
    frames = {frame.name: frame for frame in iter_frame_specs()}
    assert frames["open_file"].codon == Codon(0x0C, 0x03, 0x20)
    assert frames["open_file"].matches({"open", "file"})
