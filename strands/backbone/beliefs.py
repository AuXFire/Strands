"""Inform-turn belief extraction (BDRM §3.3, §2.3 volatility).

When the user makes a declarative statement ("Cats are cute.", "Dogs
can swim.", "Paris is a city."), we want to lift it into a structured
fact that the rest of the pipeline can use:

  - elaboration walks can surface user-stated beliefs alongside
    backbone edges
  - the Compute Module sees user beliefs in its Conditioning so it
    doesn't contradict them
  - future durable storage (per spec §2.3) can persist beliefs
    flagged with appropriate volatility

This module is intentionally narrow: pattern-match a small set of
common declarative templates against the prompt + tokens, lookup
subject/target nodes in the backbone, and emit a Belief record. We
do NOT try to handle every English declarative — only the patterns
where extraction is high-confidence.

Patterns covered (case-insensitive on the raw prompt):
  '<X> is <Y>'         → HAS_PROPERTY (when Y is adjective-like)
                       → HYPERNYM     (when Y is noun-like, esp. "a/an Y")
  '<X> are <Y>'        → same as above
  '<X> can <Y>'        → CAPABLE_OF
  '<X> have <Y>'       → HAS_A
  '<X> has <Y>'        → HAS_A
  '<X> live(s) in <Y>' → AT_LOCATION
  '<X> cause(s) <Y>'   → CAUSES

False negatives are fine — anything we can't match falls through to
the existing inform_acknowledgment template. False positives are
worse, so the rules are conservative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from strands.backbone.loader import Backbone
from strands.backbone.schema import ConceptType, Rel


@dataclass(slots=True)
class Belief:
    """One user-stated fact, the session equivalent of a backbone edge.

    ``subject_node_id``/``target_node_id`` are -1 when the corresponding
    surface couldn't be resolved to a backbone node — preserve the raw
    surface so the Compute Module can still see the assertion.
    """
    subject_surface: str
    subject_node_id: int
    relation: Rel
    target_surface: str
    target_node_id: int
    raw_prompt: str
    turn_index: int = 0


# Patterns are tried in order. Each is (regex, relation). Capture
# group 1 = subject surface; group 2 = target surface. The patterns
# anchor at sentence-ish boundaries so we don't match mid-clause.
_PATTERNS: tuple[tuple[re.Pattern[str], Rel], ...] = (
    (re.compile(r"^([a-z][a-z' ]*?) (?:is|are) (?:a |an )(.+?)[.!?]?$"),
     Rel.HYPERNYM),
    (re.compile(r"^([a-z][a-z' ]*?) (?:is|are) (.+?)[.!?]?$"),
     Rel.HAS_PROPERTY),
    (re.compile(r"^([a-z][a-z' ]*?) can (.+?)[.!?]?$"),
     Rel.CAPABLE_OF),
    (re.compile(r"^([a-z][a-z' ]*?) (?:have|has) (.+?)[.!?]?$"),
     Rel.HAS_A),
    (re.compile(r"^([a-z][a-z' ]*?) live(?:s)? in (.+?)[.!?]?$"),
     Rel.AT_LOCATION),
    (re.compile(r"^([a-z][a-z' ]*?) cause(?:s)? (.+?)[.!?]?$"),
     Rel.CAUSES),
)


# Subjects we never lift — pronouns and "I/you" are out of scope for
# now (would need coreference + speaker model). Filler determiners
# get stripped from both subject and target before lookup.
_SKIP_SUBJECTS = frozenset({
    "i", "you", "we", "they", "he", "she", "it", "this", "that",
    "these", "those",
    # Wh-words — guard against question forms that survive the early
    # interrogative check (e.g. embedded clauses).
    "what", "where", "when", "who", "why", "how", "which",
})
_LEADING_DETERMINERS = ("the ", "a ", "an ", "some ", "all ", "every ")
_INTERROGATIVE_STARTS = (
    "what ", "where ", "when ", "who ", "why ", "how ", "which ",
    "is ", "are ", "do ", "does ", "did ", "can ", "could ",
    "will ", "would ", "should ",
)


def _strip_determiners(s: str) -> str:
    s = s.strip()
    for det in _LEADING_DETERMINERS:
        if s.startswith(det):
            return s[len(det):]
    return s


def _resolve(backbone: Backbone, surface: str) -> int:
    """Best-effort lookup of a surface to a backbone node ID. Tries the
    surface, then with trailing 's' stripped (cheap depluralization).
    Returns -1 if not found."""
    candidates = backbone.nodes_for_lemma(surface)
    if not candidates and surface.endswith("s"):
        candidates = backbone.nodes_for_lemma(surface[:-1])
    if not candidates:
        return -1
    # Prefer ENTITY/EVENT senses over abstract ones.
    for nid in candidates:
        type_byte = int(backbone.nodes[nid]["concept_type"])
        if type_byte & (ConceptType.ENTITY | ConceptType.EVENT):
            return nid
    return candidates[0]


def extract_belief(
    backbone: Backbone, prompt: str, *, turn_index: int = 0,
) -> Belief | None:
    """Try to lift a declarative prompt into a Belief. Returns None
    when no high-confidence pattern matches."""
    s = prompt.strip().lower()
    # Reject obvious questions — wh-starts, aux-starts, or '?'-ends.
    if s.endswith("?"):
        return None
    if any(s.startswith(start) for start in _INTERROGATIVE_STARTS):
        return None
    for pattern, rel in _PATTERNS:
        m = pattern.match(s)
        if m is None:
            continue
        subject = _strip_determiners(m.group(1))
        target = _strip_determiners(m.group(2))
        if not subject or not target:
            continue
        if subject in _SKIP_SUBJECTS:
            continue
        # The HAS_PROPERTY pattern is permissive — only accept when
        # the target is a single token (most reliable adjective shape).
        if rel is Rel.HAS_PROPERTY and " " in target:
            continue
        subject_id = _resolve(backbone, subject)
        target_id = _resolve(backbone, target)
        return Belief(
            subject_surface=subject,
            subject_node_id=subject_id,
            relation=rel,
            target_surface=target,
            target_node_id=target_id,
            raw_prompt=prompt,
            turn_index=turn_index,
        )
    return None
