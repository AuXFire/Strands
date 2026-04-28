"""Social (SO) — groups, governance, institutions, commerce."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: groups
    (0, 0x00, ["group", "team"]),
    (0, 0x01, ["community"]),
    (0, 0x02, ["society"]),
    (0, 0x03, ["crowd", "mob"]),
    (0, 0x04, ["club", "association"]),
    (0, 0x05, ["tribe"]),
    (0, 0x06, ["organization"]),
    # Category 1: governance
    (1, 0x00, ["government"]),
    (1, 0x01, ["politics"]),
    (1, 0x02, ["law", "rule", "regulation"]),
    (1, 0x03, ["court", "justice"]),
    (1, 0x04, ["election", "vote"]),
    (1, 0x05, ["war", "warfare"]),
    (1, 0x06, ["peace"]),
    (1, 0x07, ["army", "military"]),
    (1, 0x08, ["nation", "state"]),
    # Category 2: institutions
    (2, 0x00, ["school"]),
    (2, 0x01, ["college", "university"]),
    (2, 0x02, ["hospital"]),
    (2, 0x03, ["library"]),
    (2, 0x04, ["museum"]),
    (2, 0x05, ["prison", "jail"]),
    (2, 0x06, ["bank"]),
    # Category 3: commerce
    (3, 0x00, ["market"]),
    (3, 0x01, ["store", "shop"]),
    (3, 0x02, ["business"]),
    (3, 0x03, ["company", "corporation", "firm"]),
    (3, 0x04, ["office"]),
    (3, 0x05, ["factory"]),
    (3, 0x06, ["industry"]),
    # Category 4: religion
    (4, 0x00, ["religion", "faith"]),
    (4, 0x01, ["god", "deity"]),
    (4, 0x02, ["prayer"]),
    (4, 0x03, ["sin"]),
]
