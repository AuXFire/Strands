"""Quality (QU) — size, speed, goodness, beauty, age, temperature, difficulty,
strength, light, cleanliness, wealth, truth, importance, safety, interest,
intelligence, texture, weight."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: size
    (0, 0x00, ["big", "large", "huge", "enormous"]),
    (0, 0x01, ["small", "little", "tiny", "minute"]),
    (0, 0x02, ["tall", "high"]),
    (0, 0x03, ["short", "low"]),
    (0, 0x04, ["wide", "broad"]),
    (0, 0x05, ["narrow", "thin"]),
    (0, 0x06, ["long"]),
    (0, 0x07, ["medium", "average"]),
    # Category 1: speed
    (1, 0x00, ["fast", "quick", "rapid", "swift"]),
    (1, 0x01, ["slow", "sluggish"]),
    # Category 2: goodness
    (2, 0x00, ["good", "fine", "nice"]),
    (2, 0x01, ["bad", "poor", "awful", "terrible"]),
    (2, 0x02, ["excellent", "superb", "great"]),
    (2, 0x03, ["mediocre", "average"]),
    # Category 3: beauty
    (3, 0x00, ["beautiful", "pretty", "lovely", "gorgeous"]),
    (3, 0x01, ["ugly", "hideous"]),
    (3, 0x02, ["handsome"]),
    (3, 0x03, ["plain", "ordinary"]),
    # Category 4: age
    (4, 0x00, ["old", "aged", "elderly"]),
    (4, 0x01, ["young", "youthful"]),
    (4, 0x02, ["new", "fresh", "novel"]),
    (4, 0x03, ["ancient", "antique"]),
    (4, 0x04, ["modern", "contemporary"]),
    # Category 5: temperature
    (5, 0x00, ["hot", "scorching"]),
    (5, 0x01, ["cold", "chilly", "freezing"]),
    (5, 0x02, ["warm", "tepid"]),
    (5, 0x03, ["cool"]),
    # Category 6: difficulty
    (6, 0x00, ["easy", "simple", "effortless"]),
    (6, 0x01, ["hard", "difficult", "tough"]),
    # Category 7: strength
    (7, 0x00, ["strong", "powerful", "mighty"]),
    (7, 0x01, ["weak", "feeble", "frail"]),
    (7, 0x02, ["sturdy", "robust"]),
    (7, 0x03, ["fragile", "delicate"]),
    # Category 8: light
    (8, 0x00, ["bright", "luminous", "radiant"]),
    (8, 0x01, ["dark", "dim", "gloomy"]),
    (8, 0x02, ["dull"]),
    (8, 0x03, ["shiny", "glossy"]),
    # Category 9: cleanliness
    (9, 0x00, ["clean", "spotless", "pristine"]),
    (9, 0x01, ["dirty", "filthy", "grimy"]),
    (9, 0x02, ["pure"]),
    (9, 0x03, ["polluted", "contaminated"]),
    # Category 10: wealth
    (10, 0x00, ["rich", "wealthy", "affluent"]),
    (10, 0x01, ["poor", "destitute", "impoverished"]),
    (10, 0x02, ["expensive", "costly"]),
    (10, 0x03, ["cheap", "inexpensive"]),
    # Category 11: truth
    (11, 0x00, ["true", "correct", "accurate"]),
    (11, 0x01, ["false", "untrue", "wrong"]),
    (11, 0x02, ["real", "actual", "genuine"]),
    (11, 0x03, ["fake", "phony", "counterfeit"]),
    # Category 12: importance
    (12, 0x00, ["important", "significant", "crucial"]),
    (12, 0x01, ["trivial", "unimportant"]),
    (12, 0x02, ["essential", "vital", "necessary"]),
    (12, 0x03, ["optional", "voluntary"]),
    # Category 13: safety
    (13, 0x00, ["safe", "secure"]),
    (13, 0x01, ["dangerous", "risky", "perilous"]),
    # Category 14: interest
    (14, 0x00, ["interesting", "fascinating", "engaging"]),
    (14, 0x01, ["boring", "tedious", "dull"]),
    (14, 0x02, ["funny", "humorous", "amusing"]),
    (14, 0x03, ["serious", "grave"]),
    # Category 15: intelligence
    (15, 0x00, ["smart", "intelligent", "clever", "bright"]),
    (15, 0x01, ["dumb", "stupid", "foolish"]),
    (15, 0x02, ["wise", "sage"]),
    # Category 16: texture
    (16, 0x00, ["smooth", "sleek"]),
    (16, 0x01, ["rough", "coarse"]),
    (16, 0x02, ["soft", "tender"]),
    (16, 0x03, ["hard", "firm", "rigid"]),
    # Category 17: weight
    (17, 0x00, ["heavy", "weighty"]),
    (17, 0x01, ["light", "weightless"]),
]
