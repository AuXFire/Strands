"""Module (MD) — import/export/declaration/scope/closure/dependency."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: declaration
    (0, 0x00, ["function", "fn", "func", "def"]),
    (0, 0x01, ["method"]),
    (0, 0x02, ["lambda", "arrow"]),
    (0, 0x03, ["class"]),
    (0, 0x04, ["module"]),
    (0, 0x05, ["package"]),
    (0, 0x06, ["crate"]),
    (0, 0x07, ["namespace"]),
    # Category 1: import/export
    (1, 0x00, ["import"]),
    (1, 0x01, ["export"]),
    (1, 0x02, ["require"]),
    (1, 0x03, ["include"]),
    (1, 0x04, ["use"]),
    (1, 0x05, ["from"]),
    # Category 2: scope
    (2, 0x00, ["scope"]),
    (2, 0x01, ["closure"]),
    (2, 0x02, ["context"]),
    (2, 0x03, ["binding"]),
    # Category 3: dependency
    (3, 0x00, ["dependency"]),
    (3, 0x01, ["injection"]),
    (3, 0x02, ["registry"]),
    (3, 0x03, ["resolution"]),
    (3, 0x04, ["library"]),
    (3, 0x05, ["framework"]),
]
