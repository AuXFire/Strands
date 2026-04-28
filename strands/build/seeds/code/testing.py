"""Testing (TE) — unit, integration, coverage, benchmark."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: unit
    (0, 0x00, ["test"]),
    (0, 0x01, ["assert"]),
    (0, 0x02, ["expect"]),
    (0, 0x03, ["mock"]),
    (0, 0x04, ["stub"]),
    (0, 0x05, ["spy"]),
    (0, 0x06, ["fake"]),
    (0, 0x07, ["fixture"]),
    # Category 1: integration
    (1, 0x00, ["setup"]),
    (1, 0x01, ["teardown"]),
    (1, 0x02, ["suite"]),
    (1, 0x03, ["describe"]),
    (1, 0x04, ["it"]),
    (1, 0x05, ["before", "beforeeach"]),
    (1, 0x06, ["after", "aftereach"]),
    # Category 2: coverage
    (2, 0x00, ["coverage"]),
    (2, 0x01, ["line"]),
    (2, 0x02, ["branch"]),
    (2, 0x03, ["statement"]),
    # Category 3: benchmark
    (3, 0x00, ["bench", "benchmark"]),
    (3, 0x01, ["perf", "performance"]),
    (3, 0x02, ["profile"]),
    (3, 0x03, ["measure"]),
    (3, 0x04, ["sample"]),
]
