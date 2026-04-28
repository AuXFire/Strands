"""Pattern (PT) — creational, structural, behavioral, architectural."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: creational
    (0, 0x00, ["factory"]),
    (0, 0x01, ["builder"]),
    (0, 0x02, ["singleton"]),
    (0, 0x03, ["prototype"]),
    (0, 0x04, ["pool"]),
    # Category 1: structural
    (1, 0x00, ["adapter"]),
    (1, 0x01, ["bridge"]),
    (1, 0x02, ["composite"]),
    (1, 0x03, ["decorator"]),
    (1, 0x04, ["facade"]),
    (1, 0x05, ["proxy"]),
    (1, 0x06, ["flyweight"]),
    # Category 2: behavioral
    (2, 0x00, ["observer", "subscriber"]),
    (2, 0x01, ["strategy"]),
    (2, 0x02, ["command"]),
    (2, 0x03, ["iterator"]),
    (2, 0x04, ["mediator"]),
    (2, 0x05, ["state"]),
    (2, 0x06, ["visitor"]),
    (2, 0x07, ["chain"]),
    # Category 3: architectural
    (3, 0x00, ["mvc"]),
    (3, 0x01, ["mvvm"]),
    (3, 0x02, ["repository"]),
    (3, 0x03, ["middleware"]),
    (3, 0x04, ["pipeline"]),
    (3, 0x05, ["plugin"]),
    (3, 0x06, ["pubsub", "publish", "subscribe"]),
    (3, 0x07, ["cqrs"]),
    (3, 0x08, ["eventdriven"]),
]
