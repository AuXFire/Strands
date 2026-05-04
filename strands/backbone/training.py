"""Training data plumbing for the Compute Module (BDRM §7).

Two pieces:

1. ``conditioning_to_dict()`` — pure-Python dict serialization of a
   Conditioning payload. JSON-friendly; numpy types are coerced. The
   round-trip ``dict_to_conditioning()`` exists primarily for tests
   and offline replay; it doesn't reconstruct backbone-pointing IDs
   (the NN doesn't need them).

2. ``RecordingComputeModule`` — a ComputeModule implementation that
   captures every Conditioning it sees, alongside the deterministic
   answer that was about to be returned, into a JSONL file. Used to
   collect training data: drive the deterministic system over a
   corpus of prompts, and the file becomes a stream of
   (input, candidate output) examples ready for fine-tuning a real NN.

Together these make the integration seam *observable* — you can
record sessions, replay them, and verify training data shape
without standing up a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands.backbone.compute_module import (
    AnchorFact,
    BeliefRecord,
    Conditioning,
    Fact,
    HistoryTurn,
    SpeechActTag,
)


# --- Serialization ------------------------------------------------------


def _fact_to_dict(f: Fact) -> dict[str, Any]:
    return {"relation": f.relation, "target": f.target_lemma, "weight": f.weight}


def _anchor_to_dict(a: AnchorFact) -> dict[str, Any]:
    return {
        "node_id": int(a.node_id),
        "lemmas": list(a.lemmas),
        "gloss": a.gloss,
        "hypernym": a.hypernym_lemma,
        "activation": float(a.activation),
        "facts": [_fact_to_dict(f) for f in a.facts],
    }


def _history_to_dict(h: HistoryTurn) -> dict[str, Any]:
    return {
        "turn": int(h.turn_index),
        "prompt": h.prompt,
        "response": h.response,
        "intent": h.intent,
        "question_type": h.question_type,
        "anchor_lemma": h.anchor_lemma,
        "pronoun_resolved": bool(h.pronoun_resolved),
    }


def _belief_to_dict(b: BeliefRecord) -> dict[str, Any]:
    return {
        "subject": b.subject_lemma,
        "relation": b.relation,
        "target": b.target_lemma,
        "raw": b.raw,
        "turn": int(b.turn_index),
    }


def _speech_act_to_dict(s: SpeechActTag | None) -> dict[str, Any] | None:
    if s is None:
        return None
    return {
        "act": s.act,
        "subtype": s.subtype,
        "polarity": int(s.polarity),
        "hedged": bool(s.hedged),
        "has_negation": bool(s.has_negation),
        "sentiment": int(s.sentiment),
    }


def conditioning_to_dict(c: Conditioning) -> dict[str, Any]:
    """Render a Conditioning as a JSON-serializable dict. All node
    ids are kept (callers may want them for later debugging) but
    nothing requires the backbone to interpret them."""
    return {
        "prompt": c.prompt,
        "intent": c.intent,
        "primary_anchor": (
            _anchor_to_dict(c.primary_anchor)
            if c.primary_anchor is not None
            else None
        ),
        "related_anchors": [_anchor_to_dict(a) for a in c.related_anchors],
        "deterministic_answer": c.deterministic_answer,
        "deterministic_confidence": float(c.deterministic_confidence),
        "unknowns": list(c.unknowns),
        "history": [_history_to_dict(h) for h in c.history],
        "user_beliefs": [_belief_to_dict(b) for b in c.user_beliefs],
        "speech_act": _speech_act_to_dict(c.speech_act),
    }


def dict_to_conditioning(d: dict[str, Any]) -> Conditioning:
    """Inverse of conditioning_to_dict. Reconstructs a Conditioning
    from its dict form. Used in tests and for offline replay; not
    expected to round-trip through training so it stays simple."""
    def anchor_from(a: dict[str, Any] | None) -> AnchorFact | None:
        if a is None:
            return None
        return AnchorFact(
            node_id=int(a["node_id"]),
            lemmas=list(a["lemmas"]),
            gloss=a.get("gloss", ""),
            hypernym_lemma=a.get("hypernym", ""),
            activation=float(a.get("activation", 0.0)),
            facts=[
                Fact(
                    relation=f["relation"],
                    target_lemma=f["target"],
                    weight=float(f.get("weight", 0.0)),
                ) for f in a.get("facts", [])
            ],
        )

    sa = d.get("speech_act")
    speech_act = (
        SpeechActTag(
            act=sa["act"], subtype=sa["subtype"],
            polarity=int(sa["polarity"]), hedged=bool(sa["hedged"]),
            has_negation=bool(sa["has_negation"]),
            sentiment=int(sa["sentiment"]),
        ) if sa is not None else None
    )

    return Conditioning(
        prompt=d["prompt"],
        intent=d["intent"],
        primary_anchor=anchor_from(d.get("primary_anchor")),
        related_anchors=[
            anchor_from(a) for a in d.get("related_anchors", [])
            if a is not None
        ],
        deterministic_answer=d.get("deterministic_answer", ""),
        deterministic_confidence=float(d.get("deterministic_confidence", 0.0)),
        unknowns=list(d.get("unknowns", [])),
        history=[
            HistoryTurn(
                turn_index=int(h["turn"]),
                prompt=h["prompt"],
                response=h["response"],
                intent=h.get("intent", ""),
                question_type=h.get("question_type", ""),
                anchor_lemma=h.get("anchor_lemma", ""),
                pronoun_resolved=bool(h.get("pronoun_resolved", False)),
            ) for h in d.get("history", [])
        ],
        user_beliefs=[
            BeliefRecord(
                subject_lemma=b["subject"],
                relation=b["relation"],
                target_lemma=b["target"],
                raw=b.get("raw", ""),
                turn_index=int(b.get("turn", 0)),
            ) for b in d.get("user_beliefs", [])
        ],
        speech_act=speech_act,
    )


# --- Recording compute module ------------------------------------------


@dataclass(slots=True)
class TrainingExample:
    """One captured (Conditioning, deterministic_answer) pair plus
    the override the Compute Module returned (or None when it
    deferred). This is the unit of training data."""
    conditioning: dict[str, Any]
    deterministic_answer: str
    deterministic_confidence: float
    override: str | None = None


class RecordingComputeModule:
    """A ComputeModule that captures everything it sees and writes to
    a JSONL file. Use it to collect training data without standing
    up a real NN: wrap the deterministic system, drive it over your
    prompt corpus, and the resulting file is a stream of training
    examples.

    ``inner`` is an optional wrapped Compute Module — if provided,
    its output is recorded alongside the conditioning. If None,
    the recorder defers to the deterministic answer (returns None)
    so the captured examples reflect pure deterministic behavior.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        inner: "object | None" = None,
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self.inner = inner
        self.flush_every = flush_every
        self.count = 0
        self.examples: list[TrainingExample] = []

    def complete(self, conditioning: Conditioning) -> str | None:
        override: str | None = None
        if self.inner is not None:
            override = self.inner.complete(conditioning)
        ex = TrainingExample(
            conditioning=conditioning_to_dict(conditioning),
            deterministic_answer=conditioning.deterministic_answer,
            deterministic_confidence=conditioning.deterministic_confidence,
            override=override,
        )
        self.examples.append(ex)
        self._fh.write(json.dumps({
            "conditioning": ex.conditioning,
            "deterministic_answer": ex.deterministic_answer,
            "deterministic_confidence": ex.deterministic_confidence,
            "override": ex.override,
        }, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % self.flush_every == 0:
            self._fh.flush()
        return override

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RecordingComputeModule":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
