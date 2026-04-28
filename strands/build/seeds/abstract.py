"""Abstract (AB) — mental(ideas/plans/problems), truth/fiction, power/freedom,
achievement, existence, knowledge, chance/chaos."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: mental
    (0, 0x00, ["idea", "concept", "notion", "thought"]),
    (0, 0x01, ["plan", "scheme", "strategy"]),
    (0, 0x02, ["problem", "issue", "dilemma"]),
    (0, 0x03, ["solution", "answer"]),
    (0, 0x04, ["question", "query"]),
    (0, 0x05, ["theory", "hypothesis"]),
    (0, 0x06, ["opinion", "view", "perspective"]),
    (0, 0x07, ["belief", "faith"]),
    # Category 1: truth/fiction
    (1, 0x00, ["truth", "fact", "reality"]),
    (1, 0x01, ["lie", "falsehood"]),
    (1, 0x02, ["fiction", "fantasy"]),
    (1, 0x03, ["myth", "legend"]),
    # Category 2: power/freedom
    (2, 0x00, ["power", "authority", "might"]),
    (2, 0x01, ["freedom", "liberty"]),
    (2, 0x02, ["control", "command"]),
    (2, 0x03, ["slavery", "bondage"]),
    (2, 0x04, ["right", "privilege"]),
    (2, 0x05, ["duty", "obligation"]),
    # Category 3: achievement
    (3, 0x00, ["success", "achievement", "victory"]),
    (3, 0x01, ["failure", "defeat"]),
    (3, 0x02, ["goal", "aim", "objective"]),
    (3, 0x03, ["progress", "advancement"]),
    # Category 4: existence
    (4, 0x00, ["life", "existence"]),
    (4, 0x01, ["death", "demise", "mortality"]),
    (4, 0x02, ["birth"]),
    (4, 0x03, ["being", "essence"]),
    # Category 5: knowledge
    (5, 0x00, ["knowledge", "wisdom"]),
    (5, 0x01, ["ignorance"]),
    (5, 0x02, ["skill", "ability", "talent"]),
    (5, 0x03, ["education", "learning"]),
    (5, 0x04, ["science"]),
    (5, 0x05, ["art"]),
    # Category 6: chance/chaos
    (6, 0x00, ["chance", "luck", "fortune"]),
    (6, 0x01, ["fate", "destiny"]),
    (6, 0x02, ["chaos", "disorder"]),
    (6, 0x03, ["order", "structure"]),
]
