from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from strands.codon import Codon


class RelationType(IntEnum):
    """Native relation labels carried inside v3 strand entries."""

    RELATED = 0
    SYNONYM = 1
    ISA = 2
    PART = 3
    PROPERTY = 4
    USED_FOR = 5
    CAUSES = 6
    ENTAILS = 7
    CONTEXT = 8
    ANTONYM = 9
    ROLE = 10
    ASSOCIATED = 11
    TOPIC = 12


RELATION_ALIASES: dict[str, RelationType] = {
    "RELATED": RelationType.RELATED,
    "REL": RelationType.RELATED,
    "SYN": RelationType.SYNONYM,
    "SYNONYM": RelationType.SYNONYM,
    "ISA": RelationType.ISA,
    "IS_A": RelationType.ISA,
    "PART": RelationType.PART,
    "PART_OF": RelationType.PART,
    "PROP": RelationType.PROPERTY,
    "PROPERTY": RelationType.PROPERTY,
    "USE": RelationType.USED_FOR,
    "USED_FOR": RelationType.USED_FOR,
    "CAUSE": RelationType.CAUSES,
    "CAUSES": RelationType.CAUSES,
    "ENTAIL": RelationType.ENTAILS,
    "ENTAILS": RelationType.ENTAILS,
    "CTX": RelationType.CONTEXT,
    "CONTEXT": RelationType.CONTEXT,
    "ANTI": RelationType.ANTONYM,
    "ANTONYM": RelationType.ANTONYM,
    "ROLE": RelationType.ROLE,
    "ASSOC": RelationType.ASSOCIATED,
    "ASSOCIATED": RelationType.ASSOCIATED,
    "NB": RelationType.ASSOCIATED,
    "NUMBERBATCH": RelationType.ASSOCIATED,
    "TOPIC": RelationType.TOPIC,
    "TOPICAL": RelationType.TOPIC,
}


class RelationDirection(IntEnum):
    UNDIRECTED = 0
    OUT = 1
    IN = 2


@dataclass(frozen=True, slots=True)
class TypedRelation:
    codon: Codon
    weight: int
    relation: RelationType = RelationType.RELATED
    direction: RelationDirection = RelationDirection.UNDIRECTED

    def clamped_weight(self) -> int:
        return max(0, min(255, int(self.weight)))


def parse_relation_type(value: object) -> RelationType:
    if isinstance(value, RelationType):
        return value
    if isinstance(value, int):
        try:
            return RelationType(value)
        except ValueError:
            return RelationType.RELATED
    if isinstance(value, str):
        return RELATION_ALIASES.get(value.upper(), RelationType.RELATED)
    return RelationType.RELATED


def parse_relation_direction(value: object) -> RelationDirection:
    if isinstance(value, RelationDirection):
        return value
    if isinstance(value, int):
        try:
            return RelationDirection(value)
        except ValueError:
            return RelationDirection.UNDIRECTED
    if isinstance(value, str):
        key = value.upper()
        if key in {"OUT", "FORWARD", "SRC_TO_TGT"}:
            return RelationDirection.OUT
        if key in {"IN", "REVERSE", "TGT_TO_SRC"}:
            return RelationDirection.IN
    return RelationDirection.UNDIRECTED
