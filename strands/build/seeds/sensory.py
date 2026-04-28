"""Sensory (SN) — colors, textures, tastes, sounds."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: warm colors
    (0, 0x00, ["red"]),
    (0, 0x01, ["orange"]),
    (0, 0x02, ["yellow"]),
    (0, 0x03, ["pink"]),
    (0, 0x04, ["brown"]),
    # Category 1: cool colors
    (1, 0x00, ["blue"]),
    (1, 0x01, ["green"]),
    (1, 0x02, ["purple", "violet"]),
    # Category 2: neutral colors
    (2, 0x00, ["black"]),
    (2, 0x01, ["white"]),
    (2, 0x02, ["gray", "grey"]),
    (2, 0x03, ["color", "colour", "hue"]),
    # Category 3: textures
    (3, 0x00, ["smooth"]),
    (3, 0x01, ["rough"]),
    (3, 0x02, ["soft"]),
    (3, 0x03, ["hard"]),
    (3, 0x04, ["sticky"]),
    (3, 0x05, ["slippery"]),
    (3, 0x06, ["fuzzy"]),
    # Category 4: tastes
    (4, 0x00, ["sweet"]),
    (4, 0x01, ["sour"]),
    (4, 0x02, ["bitter"]),
    (4, 0x03, ["salty"]),
    (4, 0x04, ["spicy"]),
    (4, 0x05, ["bland"]),
    (4, 0x06, ["delicious", "tasty"]),
    # Category 5: smells
    (5, 0x00, ["fragrant", "aromatic"]),
    (5, 0x01, ["stink", "stinky", "smelly"]),
    (5, 0x02, ["odor", "smell", "scent"]),
    # Category 6: sounds
    (6, 0x00, ["loud", "noisy"]),
    (6, 0x01, ["quiet", "silent"]),
    (6, 0x02, ["sound", "noise"]),
    (6, 0x03, ["music"]),
    (6, 0x04, ["song"]),
    (6, 0x05, ["bang", "boom"]),
    (6, 0x06, ["echo"]),
    (6, 0x07, ["rhythm", "beat"]),
]
