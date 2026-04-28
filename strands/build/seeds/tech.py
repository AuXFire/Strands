"""Tech (TC) — software, web, AI/security, data/hardware."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: software
    (0, 0x00, ["software", "program", "application"]),
    (0, 0x01, ["code"]),
    (0, 0x02, ["algorithm"]),
    (0, 0x03, ["database"]),
    (0, 0x04, ["bug"]),
    (0, 0x05, ["feature"]),
    (0, 0x06, ["function"]),
    (0, 0x07, ["library"]),
    (0, 0x08, ["framework"]),
    # Category 1: web
    (1, 0x00, ["internet", "web"]),
    (1, 0x01, ["website", "site"]),
    (1, 0x02, ["network"]),
    (1, 0x03, ["server"]),
    (1, 0x04, ["browser"]),
    (1, 0x05, ["email"]),
    # Category 2: AI/security
    (2, 0x00, ["intelligence"]),
    (2, 0x01, ["robot"]),
    (2, 0x02, ["password"]),
    (2, 0x03, ["security"]),
    (2, 0x04, ["encryption"]),
    (2, 0x05, ["virus", "malware"]),
    # Category 3: data/hardware
    (3, 0x00, ["data", "information"]),
    (3, 0x01, ["file"]),
    (3, 0x02, ["memory"]),
    (3, 0x03, ["storage"]),
    (3, 0x04, ["processor", "cpu"]),
    (3, 0x05, ["chip"]),
    (3, 0x06, ["device"]),
    (3, 0x07, ["machine"]),
]
