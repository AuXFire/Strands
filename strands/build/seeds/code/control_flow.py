"""Control Flow (CF) — sequential, conditional, loop, recursion, branching,
async, concurrency."""

SEEDS: list[tuple[int, int, list[str]]] = [
    # Category 0: sequential
    (0, 0x00, ["sequence", "block", "statement"]),
    (0, 0x01, ["expression"]),
    # Category 1: conditional
    (1, 0x00, ["if", "when"]),
    (1, 0x01, ["else", "elif", "elsif", "otherwise"]),
    (1, 0x02, ["switch", "case", "match"]),
    (1, 0x03, ["ternary", "select"]),
    (1, 0x04, ["unless", "guard"]),
    # Category 2: loop
    (2, 0x00, ["for", "foreach"]),
    (2, 0x01, ["while", "until"]),
    (2, 0x02, ["do"]),
    (2, 0x03, ["loop", "iterate", "iteration"]),
    (2, 0x04, ["repeat"]),
    (2, 0x05, ["map", "each"]),
    # Category 3: recursion
    (3, 0x00, ["recurse", "recursion", "recursive"]),
    # Category 4: branching
    (4, 0x00, ["break", "exit"]),
    (4, 0x01, ["continue", "next", "skip"]),
    (4, 0x02, ["return"]),
    (4, 0x03, ["throw", "raise"]),
    (4, 0x04, ["goto", "jump"]),
    (4, 0x05, ["yield"]),
    (4, 0x06, ["pass"]),
    # Category 5: async
    (5, 0x00, ["async", "asynchronous"]),
    (5, 0x01, ["await"]),
    (5, 0x02, ["promise", "future", "deferred"]),
    (5, 0x03, ["callback"]),
    (5, 0x04, ["channel"]),
    (5, 0x05, ["coroutine", "generator"]),
    # Category 6: concurrency
    (6, 0x00, ["thread", "goroutine"]),
    (6, 0x01, ["process", "fork"]),
    (6, 0x02, ["lock", "mutex"]),
    (6, 0x03, ["semaphore"]),
    (6, 0x04, ["atomic"]),
    (6, 0x05, ["spawn"]),
    (6, 0x06, ["join", "wait"]),
    (6, 0x07, ["barrier"]),
]
