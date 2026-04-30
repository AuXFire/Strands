from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from strands.codon import Codon
from strands.relations import RelationType, parse_relation_type


BASE_RELATION_SCALE: dict[RelationType, float] = {
    RelationType.RELATED: 0.85,
    RelationType.SYNONYM: 0.92,
    RelationType.ISA: 0.76,
    RelationType.PART: 0.68,
    RelationType.PROPERTY: 0.70,
    RelationType.USED_FOR: 0.72,
    RelationType.CAUSES: 0.66,
    RelationType.ENTAILS: 0.74,
    RelationType.CONTEXT: 0.58,
    RelationType.ROLE: 0.78,
    RelationType.ASSOCIATED: 0.42,
    RelationType.TOPIC: 0.35,
}


@dataclass(frozen=True, slots=True)
class ScoringProfile:
    relation_scale: dict[RelationType, float]
    antonym_penalty: float = 0.15
    query_coverage: bool = False
    symmetric_coverage: bool = False
    lexical_weight: float = 0.0
    relation_multiplier: float = 1.0
    relation_hierarchy_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class FrameSpec:
    name: str
    codon: Codon
    groups: tuple[frozenset[str], ...]
    role: int = 5
    feature: int = 0x0002

    def matches(self, words: set[str]) -> bool:
        return all(group & words for group in self.groups)


@dataclass(frozen=True, slots=True)
class Adapter:
    name: str
    profile: ScoringProfile | None = None
    frames: tuple[FrameSpec, ...] = field(default_factory=tuple)


_ADAPTER_DIR = Path(__file__).parent / "data" / "adapters"


def _profile_from_raw(raw: dict) -> ScoringProfile:
    relation_scale = dict(BASE_RELATION_SCALE)
    for key, value in raw.get("relation_scale", {}).items():
        relation_scale[parse_relation_type(key)] = float(value)
    return ScoringProfile(
        relation_scale=relation_scale,
        antonym_penalty=float(raw.get("antonym_penalty", 0.15)),
        query_coverage=bool(raw.get("query_coverage", False)),
        symmetric_coverage=bool(raw.get("symmetric_coverage", False)),
        lexical_weight=float(raw.get("lexical_weight", 0.0)),
        relation_multiplier=float(raw.get("relation_multiplier", 1.0)),
        relation_hierarchy_weight=float(raw.get("relation_hierarchy_weight", 0.0)),
    )


def _frame_from_raw(raw: dict) -> FrameSpec:
    return FrameSpec(
        name=str(raw["name"]),
        codon=Codon.from_str(str(raw["codon"])),
        groups=tuple(frozenset(str(item).lower() for item in group) for group in raw["groups"]),
        role=int(raw.get("role", 5)),
        feature=int(raw.get("feature", 0x0002)),
    )


def _adapter_from_raw(raw: dict) -> Adapter:
    return Adapter(
        name=str(raw["name"]),
        profile=_profile_from_raw(raw["profile"]) if "profile" in raw else None,
        frames=tuple(_frame_from_raw(frame) for frame in raw.get("frames", [])),
    )


@lru_cache(maxsize=1)
def load_adapters() -> dict[str, Adapter]:
    adapters: dict[str, Adapter] = {}
    if not _ADAPTER_DIR.is_dir():
        return adapters
    for path in sorted(_ADAPTER_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            adapter = _adapter_from_raw(json.load(f))
        adapters[adapter.name] = adapter
    return adapters


def get_scoring_profile(name: str) -> ScoringProfile:
    adapter = load_adapters().get(name)
    if adapter is not None and adapter.profile is not None:
        return adapter.profile
    return ScoringProfile(relation_scale=BASE_RELATION_SCALE)


def iter_frame_specs() -> tuple[FrameSpec, ...]:
    frames: list[FrameSpec] = []
    for adapter in load_adapters().values():
        frames.extend(adapter.frames)
    return tuple(frames)
