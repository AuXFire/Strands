"""Conditioning → context string formatter for the neural Compute Module.

The deterministic system produces a rich Conditioning payload. To
feed it to a sequence model we flatten it into a single string with
explicit field markers. The format is designed to be:

  - Compact: drop fields the NN doesn't need (node IDs, weights below
    a threshold).
  - Stable: same Conditioning produces the same string, byte for byte.
  - Self-describing: field markers ([INTENT], [HISTORY], etc.) so the
    model can learn to attend to specific sections.
  - Reversible enough for debugging: you can read a context string and
    see what the NN saw.

A typical context string looks like:

  [PROMPT] What is a cat?
  [INTENT] question_answering [QTYPE] definition
  [SPEECH] question/definition pol=+1
  [ANCHOR] cat (gloss: any of several large cats…)
  [FACTS] HYPERNYM:big_cat HYPERNYM:carnivore CAPABLE_OF:hunt …
  [HISTORY] T1 user: What is a dog? | bot: Dog is a member of …
  [BELIEFS] cats:HAS_PROPERTY:cute
  [DET] Cat is any of several large cats… (conf=0.90)
  [SEP]

The model is then trained to continue with the answer it should
have produced.
"""

from __future__ import annotations

from strands.backbone.compute_module import (
    AnchorFact,
    Conditioning,
    HistoryTurn,
)


# Limits to keep context bounded on CPU.
_MAX_FACTS = 16
_MAX_RELATED = 3
_MAX_HISTORY = 4
_MAX_BELIEFS = 8
_MAX_GLOSS_CHARS = 120


def _format_anchor(a: AnchorFact, *, with_facts: bool = True) -> str:
    primary = a.lemmas[0] if a.lemmas else f"node_{a.node_id}"
    gloss = (a.gloss[:_MAX_GLOSS_CHARS] + "…") if len(a.gloss) > _MAX_GLOSS_CHARS else a.gloss
    parts = [primary]
    if gloss:
        parts.append(f"(gloss: {gloss})")
    if a.hypernym_lemma:
        parts.append(f"hyp={a.hypernym_lemma}")
    s = " ".join(parts)
    if with_facts and a.facts:
        # Compact relation triples sorted as already given (priority order).
        triples = [
            f"{f.relation}:{f.target_lemma.replace(' ', '_')}"
            for f in a.facts[:_MAX_FACTS]
        ]
        s += " [FACTS] " + " ".join(triples)
    return s


def _format_history(history: list[HistoryTurn]) -> str:
    if not history:
        return ""
    recent = history[-_MAX_HISTORY:]
    parts: list[str] = []
    for h in recent:
        # Truncate long responses so context doesn't blow up.
        resp = h.response if len(h.response) <= 120 else h.response[:117] + "…"
        prompt = h.prompt if len(h.prompt) <= 120 else h.prompt[:117] + "…"
        parts.append(f"T{h.turn_index} U:{prompt} | A:{resp}")
    return " | ".join(parts)


def format_context(c: Conditioning) -> str:
    """Render a Conditioning as a deterministic single-line context
    string for the neural model. Keep this stable — change it and you
    invalidate any trained checkpoint."""
    chunks: list[str] = []
    chunks.append(f"[PROMPT] {c.prompt.strip()}")

    qtype = (
        c.speech_act.subtype if c.speech_act is not None else ""
    )
    chunks.append(f"[INTENT] {c.intent} [QTYPE] {qtype}")

    if c.speech_act is not None:
        sa = c.speech_act
        flags = []
        if sa.has_negation:
            flags.append("neg")
        if sa.hedged:
            flags.append("hedged")
        if sa.sentiment != 0:
            flags.append(f"sent={sa.sentiment:+d}")
        flag_str = (" " + " ".join(flags)) if flags else ""
        chunks.append(
            f"[SPEECH] {sa.act}/{sa.subtype} pol={sa.polarity:+d}{flag_str}"
        )

    if c.primary_anchor is not None:
        chunks.append("[ANCHOR] " + _format_anchor(c.primary_anchor))

    if c.related_anchors:
        rels = c.related_anchors[:_MAX_RELATED]
        chunks.append(
            "[RELATED] " + " | ".join(
                _format_anchor(a, with_facts=False) for a in rels
            )
        )

    history_text = _format_history(c.history)
    if history_text:
        chunks.append(f"[HISTORY] {history_text}")

    if c.user_beliefs:
        bs = c.user_beliefs[:_MAX_BELIEFS]
        triples = " ".join(
            f"{b.subject_lemma.replace(' ', '_')}:{b.relation}:"
            f"{b.target_lemma.replace(' ', '_')}"
            for b in bs
        )
        chunks.append(f"[BELIEFS] {triples}")

    if c.unknowns:
        chunks.append(f"[UNK] {' '.join(c.unknowns[:8])}")

    if c.deterministic_answer:
        chunks.append(
            f"[DET] {c.deterministic_answer.strip()} "
            f"(conf={c.deterministic_confidence:.2f})"
        )

    return " ".join(chunks) + " [SEP]"


def format_training_example(c: Conditioning, target: str) -> tuple[str, str]:
    """For training: (context_string, target_answer).
    The target is what the model is trained to produce after [SEP]."""
    return format_context(c), target.strip()
