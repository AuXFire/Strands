"""Type System (TS) — declaration, annotation, conversion, constraint."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: declaration
    (0, 0x00, ["var", "let"]),
    (0, 0x01, ["const", "constant", "final"]),
    (0, 0x02, ["type"]),
    (0, 0x03, ["interface"]),
    (0, 0x04, ["class"]),
    (0, 0x05, ["trait"]),
    (0, 0x06, ["protocol"]),
    (0, 0x07, ["abstract"]),
    (0, 0x08, ["static"]),
    (0, 0x09, ["public", "private", "protected"]),
    # Category 1: annotation
    (1, 0x00, ["generic", "template"]),
    (1, 0x01, ["nullable"]),
    (1, 0x02, ["optional"]),
    (1, 0x03, ["readonly", "immutable"]),
    (1, 0x04, ["mutable"]),
    (1, 0x05, ["volatile"]),
    # Category 2: conversion
    (2, 0x00, ["cast"]),
    (2, 0x01, ["coerce"]),
    (2, 0x02, ["parse"]),
    (2, 0x03, ["serialize"]),
    (2, 0x04, ["deserialize"]),
    (2, 0x05, ["marshal", "unmarshal"]),
    (2, 0x06, ["convert"]),
    # Category 3: constraint
    (3, 0x00, ["extends", "inherits"]),
    (3, 0x01, ["implements"]),
    (3, 0x02, ["where"]),
    (3, 0x03, ["bound"]),
    (3, 0x04, ["impl"]),
]
