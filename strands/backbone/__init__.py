"""BDRM Semantic Backbone — the compiled, fixed-width binary knowledge graph.

The backbone is the world-knowledge layer of BDRM. It holds:
  - 128-byte concept nodes drawn from WordNet synsets (and ConceptNet
    concepts in later milestones).
  - 32-byte typed weighted edges drawn from WordNet relations
    (hypernym, hyponym, meronym, antonym, similar) and ConceptNet's
    /r/* relation taxonomy.
  - A variable-width lemma table mapping each node to its surface forms.

It is built once (via ``scripts/build_backbone.py``) and memory-mapped
at runtime.
"""

from strands.backbone.beliefs import Belief, extract_belief
from strands.backbone.yesno import YesNoAnswer, answer_yesno, is_yesno_question
from strands.backbone.compute_module import (
    AnchorFact,
    BeliefRecord,
    ComputeModule,
    Conditioning,
    Fact,
    HistoryTurn,
    SpeechActTag,
    StubComputeModule,
    build_conditioning,
)
from strands.backbone.speech_act import SpeechAct, classify_speech_act
from strands.backbone.training import (
    RecordingComputeModule,
    TrainingExample,
    conditioning_to_dict,
    dict_to_conditioning,
)
from strands.backbone.inference import (
    InferenceConfig,
    InferenceResult,
    TokenCandidate,
    classify_intent,
    disambiguate,
    extract_subgraph,
    infer,
    map_tokens_to_candidates,
    spread_activation,
    tag_uncertain,
)
from strands.backbone.loader import Backbone, BackboneNode, load
from strands.backbone.response import (
    DiscourseState,
    Response,
    TurnRecord,
    respond,
)
from strands.backbone.schema import (
    EDGE_DTYPE,
    NODE_DTYPE,
    ConceptType,
    Rel,
    Source,
)

__all__ = [
    "AnchorFact",
    "Backbone",
    "BackboneNode",
    "Belief",
    "BeliefRecord",
    "ComputeModule",
    "ConceptType",
    "Conditioning",
    "DiscourseState",
    "Fact",
    "HistoryTurn",
    "TurnRecord",
    "YesNoAnswer",
    "EDGE_DTYPE",
    "InferenceConfig",
    "InferenceResult",
    "NODE_DTYPE",
    "RecordingComputeModule",
    "Rel",
    "Response",
    "Source",
    "SpeechAct",
    "SpeechActTag",
    "StubComputeModule",
    "TokenCandidate",
    "TrainingExample",
    "build_conditioning",
    "classify_intent",
    "disambiguate",
    "answer_yesno",
    "classify_speech_act",
    "conditioning_to_dict",
    "dict_to_conditioning",
    "extract_belief",
    "is_yesno_question",
    "extract_subgraph",
    "infer",
    "load",
    "map_tokens_to_candidates",
    "respond",
    "spread_activation",
    "tag_uncertain",
]
