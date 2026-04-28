"""Communication (CM) — language/text, messages/media, meetings/events."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: language/text
    (0, 0x00, ["language"]),
    (0, 0x01, ["word", "term"]),
    (0, 0x02, ["sentence"]),
    (0, 0x03, ["paragraph"]),
    (0, 0x04, ["story", "tale"]),
    (0, 0x05, ["poem", "poetry"]),
    (0, 0x06, ["text"]),
    (0, 0x07, ["letter", "alphabet"]),
    (0, 0x08, ["name"]),
    (0, 0x09, ["title"]),
    (0, 0x0A, ["meaning", "definition"]),
    # Category 1: messages/media
    (1, 0x00, ["message"]),
    (1, 0x01, ["news"]),
    (1, 0x02, ["report"]),
    (1, 0x03, ["broadcast"]),
    (1, 0x04, ["advertisement", "ad"]),
    (1, 0x05, ["announcement"]),
    (1, 0x06, ["call"]),
    (1, 0x07, ["signal"]),
    (1, 0x08, ["sign"]),
    # Category 2: meetings/events
    (2, 0x00, ["meeting", "conference"]),
    (2, 0x01, ["party", "celebration"]),
    (2, 0x02, ["event", "occasion"]),
    (2, 0x03, ["wedding", "marriage"]),
    (2, 0x04, ["funeral"]),
    (2, 0x05, ["festival"]),
    (2, 0x06, ["concert", "show"]),
    (2, 0x07, ["movie", "film"]),
    (2, 0x08, ["theater", "play"]),
    (2, 0x09, ["game", "match"]),
]
