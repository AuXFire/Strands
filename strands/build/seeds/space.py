"""Space (SP) — places(cities/nations), positions(inside/outside/near/far/direction)."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: places (cities/nations)
    (0, 0x00, ["city", "town"]),
    (0, 0x01, ["country", "nation"]),
    (0, 0x02, ["village"]),
    (0, 0x03, ["state", "province"]),
    (0, 0x04, ["world", "globe"]),
    (0, 0x05, ["continent"]),
    (0, 0x06, ["region", "area"]),
    (0, 0x07, ["place", "location", "spot"]),
    # Category 1: positions
    (1, 0x00, ["inside", "interior", "within"]),
    (1, 0x01, ["outside", "exterior"]),
    (1, 0x02, ["near", "close", "nearby"]),
    (1, 0x03, ["far", "distant", "remote"]),
    (1, 0x04, ["above", "over"]),
    (1, 0x05, ["below", "under", "beneath"]),
    (1, 0x06, ["between"]),
    (1, 0x07, ["around"]),
    (1, 0x08, ["here"]),
    (1, 0x09, ["there"]),
    # Category 2: direction
    (2, 0x00, ["north"]),
    (2, 0x01, ["south"]),
    (2, 0x02, ["east"]),
    (2, 0x03, ["west"]),
    (2, 0x04, ["up", "upward"]),
    (2, 0x05, ["down", "downward"]),
    (2, 0x06, ["left"]),
    (2, 0x07, ["right"]),
    (2, 0x08, ["forward"]),
    (2, 0x09, ["backward"]),
    (2, 0x0A, ["direction"]),
    (2, 0x0B, ["distance"]),
    (2, 0x0C, ["middle", "center", "centre"]),
]
