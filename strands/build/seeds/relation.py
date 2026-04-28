"""Relation (RL) — togetherness, causation, similarity, difference."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: togetherness
    (0, 0x00, ["together"]),
    (0, 0x01, ["apart", "separate"]),
    (0, 0x02, ["with"]),
    (0, 0x03, ["alone"]),
    (0, 0x04, ["pair", "couple"]),
    # Category 1: causation
    (1, 0x00, ["cause"]),
    (1, 0x01, ["effect", "result"]),
    (1, 0x02, ["reason"]),
    (1, 0x03, ["because"]),
    (1, 0x04, ["consequence"]),
    (1, 0x05, ["influence"]),
    # Category 2: similarity
    (2, 0x00, ["same", "identical"]),
    (2, 0x01, ["similar", "alike"]),
    (2, 0x02, ["like"]),
    (2, 0x03, ["resemble"]),
    (2, 0x04, ["copy", "duplicate"]),
    # Category 3: difference
    (3, 0x00, ["different", "distinct"]),
    (3, 0x01, ["unlike"]),
    (3, 0x02, ["opposite", "contrary"]),
    (3, 0x03, ["unique"]),
    (3, 0x04, ["change", "alter"]),
    # Category 4: containment/parts
    (4, 0x00, ["part", "piece"]),
    (4, 0x01, ["whole"]),
    (4, 0x02, ["contain", "include"]),
    (4, 0x03, ["belong"]),
    (4, 0x04, ["have", "possess"]),
]
