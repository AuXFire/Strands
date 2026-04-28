"""Movement (MV) — rotation, vibration, flow."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: rotation
    (0, 0x00, ["rotate", "spin", "turn"]),
    (0, 0x01, ["roll"]),
    (0, 0x02, ["twist"]),
    (0, 0x03, ["circle", "loop"]),
    # Category 1: vibration/oscillation
    (1, 0x00, ["shake", "tremble", "quiver"]),
    (1, 0x01, ["vibrate"]),
    (1, 0x02, ["swing", "sway"]),
    (1, 0x03, ["bounce"]),
    (1, 0x04, ["tremor"]),
    # Category 2: flow
    (2, 0x00, ["flow", "stream"]),
    (2, 0x01, ["pour"]),
    (2, 0x02, ["drip", "drop"]),
    (2, 0x03, ["flood"]),
    (2, 0x04, ["leak"]),
    (2, 0x05, ["splash"]),
    # Category 3: linear
    (3, 0x00, ["motion", "movement"]),
    (3, 0x01, ["slide", "glide"]),
    (3, 0x02, ["drift", "float"]),
    (3, 0x03, ["wave"]),
    (3, 0x04, ["stillness", "stop"]),
]
