"""Corpus generation for the Compute Module training pipeline.

We need a stream of (Conditioning, deterministic_answer) pairs. The
generator produces varied, well-formed prompts spanning every
question class and intent the system handles, then drives the
deterministic system through them with RecordingComputeModule
capturing each turn into JSONL.

The corpus is deterministic: same seed words, same backbone, same
output. Re-running produces byte-identical training data for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from strands.backbone.loader import Backbone


# Seed lemmas with rich backbone coverage — these survive all the
# question types because they have edges across HYPERNYM,
# CAPABLE_OF, AT_LOCATION, USED_FOR, MADE_OF, HAS_PROPERTY.
_RICH_NOUNS: tuple[str, ...] = (
    "cat", "dog", "bird", "fish", "horse", "cow", "lion", "tiger",
    "rabbit", "mouse", "wolf", "bear", "elephant", "monkey",
    "apple", "banana", "tomato", "potato", "carrot", "bread",
    "water", "milk", "coffee", "tea",
    "car", "truck", "bicycle", "boat", "plane", "train",
    "house", "school", "hospital", "library", "store", "office",
    "tree", "flower", "grass", "rose", "oak",
    "book", "pencil", "table", "chair", "bed", "lamp", "door",
    "phone", "computer", "television", "camera", "clock",
    "city", "country", "river", "mountain", "ocean", "forest",
    "doctor", "teacher", "engineer", "farmer", "artist",
    "sun", "moon", "star", "earth", "sky", "cloud", "wind",
)

_PROPERTIES: tuple[str, ...] = (
    "warm", "cold", "small", "large", "soft", "hard", "fast",
    "slow", "loud", "quiet", "bright", "dark", "old", "new",
    "happy", "sad", "smart", "strong",
)

_CATEGORIES: tuple[str, ...] = (
    "animal", "mammal", "vehicle", "plant", "food", "tool",
    "person", "place", "color", "feeling",
)

_VERBS: tuple[str, ...] = (
    "fly", "swim", "run", "walk", "jump", "sing", "eat", "sleep",
    "drink", "read", "write", "draw",
)


@dataclass(frozen=True)
class GeneratedPrompt:
    """One generated prompt ready to drive through the system,
    with metadata describing what it's testing."""
    text: str
    category: str   # 'definition' | 'location' | 'ability' | 'composition' |
                    # 'purpose' | 'yesno_hypernym' | 'yesno_capable' |
                    # 'inform' | 'social' | 'elaboration' | 'pronoun'
    seeds: tuple[str, ...] = ()


def _has_lemma(backbone: Backbone, lemma: str) -> bool:
    return len(backbone.nodes_for_lemma(lemma)) > 0


def generate_prompts(
    backbone: Backbone, *, max_per_category: int | None = None,
) -> list[GeneratedPrompt]:
    """Build a varied corpus of prompts. Filters seed words against
    the backbone so we don't generate prompts the system can't anchor.
    """
    nouns = [w for w in _RICH_NOUNS if _has_lemma(backbone, w)]
    props = [w for w in _PROPERTIES if _has_lemma(backbone, w)]
    cats = [w for w in _CATEGORIES if _has_lemma(backbone, w)]
    verbs = [w for w in _VERBS if _has_lemma(backbone, w)]

    out: list[GeneratedPrompt] = []

    def cap(lst: list[GeneratedPrompt]) -> None:
        if max_per_category is None:
            out.extend(lst)
        else:
            out.extend(lst[:max_per_category])

    # Definition questions.
    cap([GeneratedPrompt(f"What is a {n}?", "definition", (n,))
         for n in nouns])
    cap([GeneratedPrompt(f"What is {n}?", "definition", (n,))
         for n in nouns[:20]])

    # Location.
    cap([GeneratedPrompt(f"Where do {n}s live?", "location", (n,))
         for n in nouns if " " not in n][:30])
    cap([GeneratedPrompt(f"Where can a {n} be found?", "location", (n,))
         for n in nouns[:20]])

    # Ability.
    cap([GeneratedPrompt(f"What can a {n} do?", "ability", (n,))
         for n in nouns[:30]])
    cap([GeneratedPrompt(f"What does a {n} do?", "ability", (n,))
         for n in nouns[:20]])

    # Composition.
    cap([GeneratedPrompt(f"What is a {n} made of?", "composition", (n,))
         for n in nouns[:25]])

    # Purpose.
    cap([GeneratedPrompt(f"What is a {n} used for?", "purpose", (n,))
         for n in nouns[:25]])

    # Yes/no — hypernym (true and false pairs).
    for n in nouns[:30]:
        for c in cats[:5]:
            cap([GeneratedPrompt(
                f"Is a {n} a {c}?",
                "yesno_hypernym",
                (n, c),
            )])

    # Yes/no — capability.
    for n in nouns[:20]:
        for v in verbs[:6]:
            cap([GeneratedPrompt(
                f"Can a {n} {v}?",
                "yesno_capable",
                (n, v),
            )])

    # Inform — beliefs.
    for n in nouns[:25]:
        for p in props[:5]:
            cap([GeneratedPrompt(f"{n.capitalize()}s are {p}", "inform", (n, p))])
    for n in nouns[:20]:
        for v in verbs[:5]:
            cap([GeneratedPrompt(f"{n.capitalize()}s can {v}", "inform", (n, v))])

    # Social.
    cap([GeneratedPrompt(p, "social")
         for p in (
             "Hello", "Hi there", "Hey", "Goodbye", "Bye",
             "Thanks for your help", "Thank you", "Sorry", "No problem",
         )])

    return out


def generate_multiturn_sessions(
    backbone: Backbone, *, n_sessions: int = 30,
) -> list[list[GeneratedPrompt]]:
    """Build a few canonical multi-turn sessions: a definition followed
    by elaborations, pronoun follow-ups, and yes/no probes. Exercises
    the discourse-state side of the pipeline so the captured
    Conditioning includes history + beliefs."""
    nouns = [w for w in _RICH_NOUNS if _has_lemma(backbone, w)]
    sessions: list[list[GeneratedPrompt]] = []
    for n in nouns[:n_sessions]:
        sessions.append([
            GeneratedPrompt(f"What is a {n}?", "definition", (n,)),
            GeneratedPrompt("Tell me more.", "elaboration"),
            GeneratedPrompt("And?", "elaboration"),
            GeneratedPrompt("What is it?", "pronoun"),
            GeneratedPrompt(f"Can a {n} run?", "yesno_capable", (n, "run")),
            GeneratedPrompt(f"Where do {n}s live?", "location", (n,)),
            GeneratedPrompt(f"{n.capitalize()}s are interesting", "inform", (n,)),
        ])
    return sessions
