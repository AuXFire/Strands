"""Quantity (QT) — totality, comparison, numbers."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: totality
    (0, 0x00, ["all", "every", "entire", "whole"]),
    (0, 0x01, ["none", "nothing", "zero"]),
    (0, 0x02, ["some", "few"]),
    (0, 0x03, ["many", "lots", "numerous"]),
    (0, 0x04, ["most"]),
    (0, 0x05, ["several"]),
    # Category 1: comparison
    (1, 0x00, ["more", "extra"]),
    (1, 0x01, ["less", "fewer"]),
    (1, 0x02, ["enough", "sufficient"]),
    (1, 0x03, ["too", "excessive"]),
    (1, 0x04, ["equal", "same"]),
    (1, 0x05, ["different", "diverse"]),
    # Category 2: numbers
    (2, 0x00, ["one"]),
    (2, 0x01, ["two", "pair", "couple"]),
    (2, 0x02, ["three"]),
    (2, 0x03, ["four"]),
    (2, 0x04, ["five"]),
    (2, 0x05, ["ten"]),
    (2, 0x06, ["hundred"]),
    (2, 0x07, ["thousand"]),
    (2, 0x08, ["million"]),
    (2, 0x09, ["half"]),
    (2, 0x0A, ["quarter"]),
    (2, 0x0B, ["dozen"]),
    # Category 3: amount
    (3, 0x00, ["amount", "quantity"]),
    (3, 0x01, ["number", "count"]),
    (3, 0x02, ["total", "sum"]),
    (3, 0x03, ["pair", "couple", "duo"]),
]
