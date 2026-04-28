"""Error (ER) — handling, validation, result, logging."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: handling
    (0, 0x00, ["try"]),
    (0, 0x01, ["catch", "except"]),
    (0, 0x02, ["finally", "ensure"]),
    (0, 0x03, ["throw", "raise"]),
    (0, 0x04, ["panic"]),
    (0, 0x05, ["recover"]),
    (0, 0x06, ["rescue"]),
    # Category 1: validation
    (1, 0x00, ["assert"]),
    (1, 0x01, ["check"]),
    (1, 0x02, ["verify"]),
    (1, 0x03, ["sanitize"]),
    (1, 0x04, ["validate"]),
    (1, 0x05, ["guard"]),
    (1, 0x06, ["expect"]),
    # Category 2: result
    (2, 0x00, ["ok", "success"]),
    (2, 0x01, ["err", "failure"]),
    (2, 0x02, ["some"]),
    (2, 0x03, ["none"]),
    (2, 0x04, ["maybe"]),
    (2, 0x05, ["either"]),
    (2, 0x06, ["result"]),
    # Category 3: logging
    (3, 0x00, ["log"]),
    (3, 0x01, ["trace"]),
    (3, 0x02, ["debug"]),
    (3, 0x03, ["info"]),
    (3, 0x04, ["warn"]),
    (3, 0x05, ["error"]),
    (3, 0x06, ["fatal", "critical"]),
    (3, 0x07, ["exception"]),
]
