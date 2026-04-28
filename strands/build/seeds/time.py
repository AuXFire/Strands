"""Time (TM) — temporal, durations, frequency, rate."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: temporal
    (0, 0x00, ["now", "present", "today"]),
    (0, 0x01, ["past", "yesterday", "before"]),
    (0, 0x02, ["future", "tomorrow", "later"]),
    (0, 0x03, ["time"]),
    # Category 1: durations
    (1, 0x00, ["moment", "instant"]),
    (1, 0x01, ["second"]),
    (1, 0x02, ["minute"]),
    (1, 0x03, ["hour"]),
    (1, 0x04, ["day"]),
    (1, 0x05, ["week"]),
    (1, 0x06, ["month"]),
    (1, 0x07, ["year"]),
    (1, 0x08, ["decade"]),
    (1, 0x09, ["century"]),
    (1, 0x0A, ["forever", "eternity"]),
    # Category 2: parts of day
    (2, 0x00, ["morning", "dawn"]),
    (2, 0x01, ["afternoon"]),
    (2, 0x02, ["evening", "dusk"]),
    (2, 0x03, ["night", "nighttime"]),
    (2, 0x04, ["noon", "midday"]),
    (2, 0x05, ["midnight"]),
    # Category 3: frequency
    (3, 0x00, ["always", "constantly"]),
    (3, 0x01, ["never"]),
    (3, 0x02, ["often", "frequently"]),
    (3, 0x03, ["rarely", "seldom"]),
    (3, 0x04, ["sometimes", "occasionally"]),
    (3, 0x05, ["usually", "typically"]),
    # Category 4: rate
    (4, 0x00, ["sudden", "abrupt"]),
    (4, 0x01, ["gradual", "slow"]),
    (4, 0x02, ["instant", "immediate"]),
    # Category 5: events/seasons
    (5, 0x00, ["spring"]),
    (5, 0x01, ["summer"]),
    (5, 0x02, ["autumn", "fall"]),
    (5, 0x03, ["winter"]),
    (5, 0x04, ["start", "begin", "commence"]),
    (5, 0x05, ["end", "finish", "conclude"]),
]
