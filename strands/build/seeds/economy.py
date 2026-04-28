"""Economy (EC) — systems, trade, pricing, investment."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: systems
    (0, 0x00, ["economy"]),
    (0, 0x01, ["capitalism"]),
    (0, 0x02, ["socialism"]),
    (0, 0x03, ["market"]),
    # Category 1: trade
    (1, 0x00, ["trade", "commerce"]),
    (1, 0x01, ["buy", "purchase"]),
    (1, 0x02, ["sell"]),
    (1, 0x03, ["exchange"]),
    (1, 0x04, ["import"]),
    (1, 0x05, ["export"]),
    # Category 2: pricing
    (2, 0x00, ["price", "cost"]),
    (2, 0x01, ["value", "worth"]),
    (2, 0x02, ["expensive"]),
    (2, 0x03, ["cheap", "inexpensive"]),
    (2, 0x04, ["free"]),
    (2, 0x05, ["discount"]),
    (2, 0x06, ["tax"]),
    # Category 3: investment
    (3, 0x00, ["invest", "investment"]),
    (3, 0x01, ["stock", "share"]),
    (3, 0x02, ["profit", "gain"]),
    (3, 0x03, ["loss", "deficit"]),
    (3, 0x04, ["interest"]),
    (3, 0x05, ["loan", "debt"]),
    (3, 0x06, ["save", "savings"]),
    (3, 0x07, ["bank"]),
    # Category 4: work/income
    (4, 0x00, ["job", "work", "employment"]),
    (4, 0x01, ["salary", "wage", "pay"]),
    (4, 0x02, ["income", "earnings"]),
    (4, 0x03, ["expense", "spending"]),
    (4, 0x04, ["budget"]),
]
