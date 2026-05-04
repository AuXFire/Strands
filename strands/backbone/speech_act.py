"""Speech-act tagging (BDRM §3.2 enhancement).

The intent classifier returns a coarse 4-way label
(question_answering / instruction / inform / social). The Compute
Module benefits from finer signals: question subtype (yesno, wh,
location, ability, …), polarity (positive vs negative assertion),
and a top-level speech act category (assertion, question, command,
expression).

This module produces a SpeechAct dataclass that gets attached to
Conditioning so the NN sees the full discourse structure of the
turn, not just the intent label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Negation cues that flip polarity. Cheap surface match — for v1 we
# don't try to handle scope (e.g. 'I don't think it's not raining').
_NEGATION_CUES = (
    " not ", " no ", " never ", " neither ", " nor ", " nothing ",
    " nobody ", " nowhere ", "n't ", "n't.", "n't,", " cannot ",
)

# Hedging cues that lower assertion confidence — useful signal for
# the NN even though we don't change the deterministic answer based
# on them.
_HEDGE_CUES = (
    "i think", "i believe", "maybe", "perhaps", "possibly",
    "might", "could be", "seems like", "i guess", "probably",
)

# Sentiment cues — cheap polarity hint.
_POSITIVE_CUES = (
    "love", "great", "awesome", "wonderful", "happy", "glad",
    "amazing", "excellent", "perfect", "thanks", "thank you",
)
_NEGATIVE_CUES = (
    "hate", "terrible", "awful", "bad", "sad", "angry",
    "disappointing", "wrong", "broken", "sorry",
)


@dataclass(slots=True)
class SpeechAct:
    """Rich tag for one turn. ``act`` is the top-level category;
    ``subtype`` refines it (e.g. act='question', subtype='yesno');
    ``polarity`` is +1 / 0 / -1 for positive / neutral / negative;
    ``hedged`` flags soft-assertion language."""
    act: str          # 'question' | 'assertion' | 'directive' | 'expressive'
    subtype: str      # 'yesno' | 'definition' | 'location' | 'greeting' | …
    polarity: int     # +1, 0, -1
    hedged: bool
    has_negation: bool
    sentiment: int    # +1, 0, -1


def _has_any(s: str, cues: tuple[str, ...]) -> bool:
    return any(c in s for c in cues)


def classify_speech_act(prompt: str, *, intent: str, question_type: str) -> SpeechAct:
    """Combine the existing intent + question_type signals with a
    few cheap surface checks (negation, hedging, sentiment) to
    produce a richer per-turn descriptor."""
    s = " " + prompt.strip().lower() + " "

    has_negation = _has_any(s, _NEGATION_CUES)
    hedged = _has_any(s, _HEDGE_CUES)

    sentiment = 0
    if _has_any(s, _POSITIVE_CUES):
        sentiment = 1
    if _has_any(s, _NEGATIVE_CUES):
        # Negation flips a positive cue if both are present
        sentiment = -1 if sentiment >= 0 else sentiment

    # Map intent → top-level act.
    if intent == "question_answering":
        act = "question"
        subtype = question_type or "definition"
    elif intent == "instruction":
        act = "directive"
        subtype = "request"
    elif intent == "social":
        act = "expressive"
        s_low = prompt.strip().lower()
        if any(g in s_low for g in ("hi", "hello", "hey")):
            subtype = "greeting"
        elif "thank" in s_low:
            subtype = "thanks"
        elif "sorry" in s_low:
            subtype = "apology"
        elif "bye" in s_low:
            subtype = "farewell"
        else:
            subtype = "phatic"
    elif intent == "inform":
        act = "assertion"
        subtype = "fact"
    else:
        act = "unknown"
        subtype = ""

    # Polarity follows negation — assertion polarity flips, question
    # polarity tracks whether the question itself is negated
    # ("Isn't a cat an animal?") which is rare; default to +.
    polarity = -1 if (act == "assertion" and has_negation) else 1
    if act == "question" and has_negation:
        polarity = -1
    return SpeechAct(
        act=act,
        subtype=subtype,
        polarity=polarity,
        hedged=hedged,
        has_negation=has_negation,
        sentiment=sentiment,
    )
