"""Data Structure (DS) — primitive, linear, associative, tree, graph, composite."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: primitive
    (0, 0x00, ["int", "integer", "long", "short", "i32", "i64", "u32", "u64"]),
    (0, 0x01, ["float", "double", "f32", "f64", "real"]),
    (0, 0x02, ["string", "str", "text", "char"]),
    (0, 0x03, ["bool", "boolean"]),
    (0, 0x04, ["byte", "u8"]),
    (0, 0x05, ["null", "nil", "none", "void", "undefined"]),
    # Category 1: linear
    (1, 0x00, ["array"]),
    (1, 0x01, ["list", "linkedlist"]),
    (1, 0x02, ["vector", "vec"]),
    (1, 0x03, ["deque"]),
    (1, 0x04, ["buffer"]),
    (1, 0x05, ["stack"]),
    (1, 0x06, ["queue"]),
    (1, 0x07, ["slice"]),
    # Category 2: associative
    (2, 0x00, ["map", "dict", "dictionary"]),
    (2, 0x01, ["set"]),
    (2, 0x02, ["hashmap"]),
    (2, 0x03, ["hashtable"]),
    (2, 0x04, ["btreemap"]),
    # Category 3: tree
    (3, 0x00, ["tree"]),
    (3, 0x01, ["binarytree", "bst"]),
    (3, 0x02, ["avl"]),
    (3, 0x03, ["btree"]),
    (3, 0x04, ["trie"]),
    (3, 0x05, ["heap"]),
    # Category 4: graph
    (4, 0x00, ["graph"]),
    (4, 0x01, ["dag"]),
    (4, 0x02, ["edge"]),
    (4, 0x03, ["node", "vertex"]),
    # Category 5: composite
    (5, 0x00, ["struct", "structure"]),
    (5, 0x01, ["record"]),
    (5, 0x02, ["tuple"]),
    (5, 0x03, ["union"]),
    (5, 0x04, ["enum", "enumeration"]),
    (5, 0x05, ["object"]),
    (5, 0x06, ["class"]),
]
