"""Response formulation + surface realization (BDRM §3.4 + §3.5).

This is the minimal end-to-end path that turns a prompt into a sentence.
Stays template-driven and deterministic for now; the Compute Module
(§4) plugs in for low-confidence cases.

Pipeline:
  1. Run prompt-to-backbone inference (M2) → InferenceResult
  2. Pick a primary anchor (the most-activated content node)
  3. Select content based on intent:
       question_answering → use gloss + hypernym/has_property facts
       inform             → acknowledge + light paraphrase
       instruction        → acknowledge + state best-effort understanding
       social             → canonical greeting/thanks reply
  4. Render to surface text via small templates
  5. Lightweight syntactic smoothing (capitalize, punctuate)

Returns a ``Response`` dataclass with the answer string, the chosen
anchors, the discourse-state delta (M3 stub), and a confidence score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from strands.backbone.compute_module import (
    ComputeModule,
    build_conditioning,
)
from strands.backbone.inference import InferenceResult, infer
from strands.backbone.loader import Backbone
from strands.backbone.schema import Rel


# Pronouns that resolve against the rolling DiscourseState topic.
# "it"/"they"/"them" are stop-words and get filtered before tokenization,
# so we look for them in the raw prompt instead.
_PRONOUNS = frozenset({
    "it", "they", "them", "this", "that", "these", "those",
})
_WORD_RE = re.compile(r"[a-z']+")


def _prompt_has_pronoun(prompt: str) -> bool:
    return any(w in _PRONOUNS for w in _WORD_RE.findall(prompt.lower()))


# Prompts that ask for additional information about the active topic.
# Matched on the raw prompt (after stripping/lowercasing) — the intent
# classifier treats these as instructions or social.
_ELABORATION_PHRASES = (
    "tell me more",
    "what else",
    "and what else",
    "go on",
    "continue",
    "anything else",
    "more about",
)


def _is_elaboration(prompt: str) -> bool:
    s = prompt.strip().lower().rstrip(".?!")
    if s in {"more", "and", "and?", "go on", "continue"}:
        return True
    return any(s.startswith(p) for p in _ELABORATION_PHRASES)


# Relation walk for topic elaboration, in priority order. Each entry is
# (relation, sentence template) where {subject} and {target} are filled
# with backbone lemmas. Templates are kept simple — surface smoothing
# (capitalize, period) happens at the call site.
_ELABORATION_RELATIONS: tuple[tuple[Rel, str], ...] = (
    (Rel.HYPERNYM,     "{subject} is a kind of {target}"),
    (Rel.HAS_PROPERTY, "{subject} is {target}"),
    (Rel.MERONYM,      "{subject} has {target} as a part"),
    (Rel.AT_LOCATION,  "{subject} can be found at {target}"),
    (Rel.USED_FOR,     "{subject} is used for {target}"),
    (Rel.CAPABLE_OF,   "{subject} is capable of {target}"),
    (Rel.MADE_OF,      "{subject} is made of {target}"),
    (Rel.HYPONYM,      "examples of {subject} include {target}"),
    (Rel.HOLONYM,      "{subject} is part of {target}"),
)


# --- Single-turn discourse state stub (M3) ------------------------------


@dataclass(slots=True)
class DiscourseState:
    """Minimal multi-turn state. Stored across calls if you keep the
    same instance; reset for fresh sessions."""
    active_topic_node_ids: list[int] = field(default_factory=list)
    entity_register: dict[str, int] = field(default_factory=dict)  # surface→node_id
    last_intent: str = ""
    turn_count: int = 0
    # Facts emitted by the elaboration walker so successive
    # 'tell me more' calls produce novel content. Key:
    # (subject_node_id, relation_id, target_node_id).
    emitted_facts: set[tuple[int, int, int]] = field(default_factory=set)

    def update(
        self, result: InferenceResult, *,
        anchor_id: int | None, backbone: Backbone | None = None,
    ) -> None:
        self.turn_count += 1
        self.last_intent = result.intent
        # Refresh the rolling topic only when this turn introduces a new
        # ENTITY-type anchor — a noun-phrase the conversation is "about".
        # Verb-anchored turns ("Tell me more.", "Show it to me.") leave
        # the previous topic intact so follow-up pronouns still resolve.
        is_entity_anchor = (
            anchor_id is not None
            and backbone is not None
            and bool(int(backbone.nodes[anchor_id]["concept_type"]) & 0x01)
        )
        if is_entity_anchor:
            top = sorted(
                result.activations.items(), key=lambda x: -x[1],
            )[:5]
            ordered = [anchor_id] + [n for n, _ in top if n != anchor_id]
            self.active_topic_node_ids = ordered[:5]
        if anchor_id is not None:
            for i, tok in enumerate(result.tokens):
                if i in result.anchors:
                    self.entity_register[tok.surface] = result.anchors[i]


# --- Response container -------------------------------------------------


@dataclass(slots=True)
class Response:
    text: str
    inference: InferenceResult
    primary_anchor_id: int | None
    confidence: float
    needs_compute_module: bool = False
    state: DiscourseState | None = None
    # True when the Compute Module ran and overrode the deterministic
    # answer. False when the deterministic answer was returned (either
    # because confidence was high enough, or because the Compute Module
    # deferred by returning None).
    compute_module_used: bool = False
    # True when the primary anchor came from pronoun resolution against
    # the discourse state rather than direct lemma lookup.
    pronoun_resolved: bool = False


# --- Helpers ------------------------------------------------------------


def _primary_anchor(
    backbone: Backbone, result: InferenceResult,
) -> int | None:
    """Pick the most-salient anchored content token. Heuristic: prefer
    ENTITY/EVENT over PROPERTY, break ties by activation."""
    if not result.anchors:
        return None
    scored: list[tuple[int, int, float]] = []  # (node_id, type_priority, activation)
    for token_idx, node_id in result.anchors.items():
        n = backbone.nodes[node_id]
        type_byte = int(n["concept_type"])
        # ENTITY=0x01 → 3, EVENT=0x02 → 2, others lower
        if type_byte & 0x01:
            priority = 3
        elif type_byte & 0x02:
            priority = 2
        elif type_byte & 0x10:  # FRAME
            priority = 1
        else:
            priority = 0
        scored.append((
            node_id,
            priority,
            result.activations.get(node_id, 0.0),
        ))
    # Highest priority, then highest activation
    scored.sort(key=lambda t: (-t[1], -t[2]))
    return scored[0][0]


def _best_lemma(backbone: Backbone, node_id: int) -> str:
    """Return the cleanest lemma for a node — shortest single-word lemma
    if available, else the first one."""
    lemmas = backbone.lemmas_for(node_id)
    if not lemmas:
        return ""
    single = [l for l in lemmas if " " not in l]
    if single:
        single.sort(key=len)
        return single[0]
    return lemmas[0]


def _hypernym_target(
    backbone: Backbone, node_id: int,
) -> tuple[int, int] | None:
    """Highest-weight HYPERNYM edge → (target_node_id, weight)."""
    edges = backbone.edges_with_relation(node_id, Rel.HYPERNYM)
    if edges.size == 0:
        return None
    best_idx = int(edges["weight"].argmax())
    return int(edges[best_idx]["target_id"]), int(edges[best_idx]["weight"])


def _article(word: str) -> str:
    """English a/an by initial vowel sound (heuristic)."""
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"


# --- Content selection per intent --------------------------------------


def _answer_question(
    backbone: Backbone, anchor_id: int,
) -> tuple[str, float]:
    """For a question, prefer the gloss; fall back to a hypernym
    sentence; final fallback is just the lemmas list. When a gloss is
    very short, append a hypernym clause so the answer carries more
    signal."""
    subject = _best_lemma(backbone, anchor_id) or "it"
    gloss = backbone.gloss_for(anchor_id)
    if gloss:
        first = gloss[0].lower() + gloss[1:] if len(gloss) > 1 else gloss
        text = f"{subject.capitalize()} is {first}."
        # Glosses under ~40 chars are often terse — fold in the hypernym
        # so the answer still reads as substantive.
        if len(gloss) < 40:
            hyp = _hypernym_target(backbone, anchor_id)
            if hyp:
                target_lemma = _best_lemma(backbone, hyp[0])
                if target_lemma and target_lemma != subject:
                    article = _article(target_lemma)
                    text = (
                        f"{subject.capitalize()} is {first}, "
                        f"{article} kind of {target_lemma}."
                    )
        return text, 0.9

    hyp = _hypernym_target(backbone, anchor_id)
    if hyp:
        target_id, weight = hyp
        target_lemma = _best_lemma(backbone, target_id) or "something"
        article = _article(target_lemma)
        confidence = min(0.85, weight / 0xFFFF + 0.4)
        return f"{subject.capitalize()} is {article} {target_lemma}.", confidence

    # Nothing useful — punt to Compute Module.
    return f"I don't have enough information about {subject}.", 0.2


def _elaborate_topic(
    backbone: Backbone, anchor_id: int,
    emitted: set[tuple[int, int, int]],
) -> tuple[str, float]:
    """Walk relations from the anchor in priority order, returning the
    first highest-weight target that hasn't been emitted yet. Records
    the chosen fact in ``emitted`` so later elaboration calls produce
    novel content."""
    subject = _best_lemma(backbone, anchor_id) or "it"
    for rel, template in _ELABORATION_RELATIONS:
        edges = backbone.edges_with_relation(anchor_id, rel)
        if edges.size == 0:
            continue
        order = np.argsort(-edges["weight"].astype(np.int64))
        for idx in order:
            target_id = int(edges[int(idx)]["target_id"])
            key = (anchor_id, int(rel), target_id)
            if key in emitted:
                continue
            target_lemma = _best_lemma(backbone, target_id)
            if not target_lemma or target_lemma == subject:
                continue
            emitted.add(key)
            sentence = template.format(subject=subject, target=target_lemma)
            return sentence[0].upper() + sentence[1:] + ".", 0.8
    # Exhausted — nothing new to say.
    return f"That's all I have about {subject}.", 0.4


def _inform_acknowledgment(
    backbone: Backbone, anchor_id: int | None,
) -> tuple[str, float]:
    if anchor_id is None:
        return "Got it.", 0.6
    subject = _best_lemma(backbone, anchor_id) or "that"
    return f"Got it — noted about {subject}.", 0.7


def _instruction_acknowledgment(
    backbone: Backbone, anchor_id: int | None,
) -> tuple[str, float]:
    if anchor_id is None:
        return "OK.", 0.5
    subject = _best_lemma(backbone, anchor_id) or "the request"
    return f"OK — I'll work on {subject}.", 0.6


def _social_response(prompt: str) -> tuple[str, float]:
    lower = prompt.strip().lower()
    if any(g in lower for g in ("hi", "hello", "hey")):
        return "Hello.", 0.95
    if "thank" in lower:
        return "You're welcome.", 0.95
    if "sorry" in lower:
        return "No worries.", 0.9
    if "bye" in lower:
        return "Goodbye.", 0.95
    return "Hello.", 0.7


# --- Top-level respond() -----------------------------------------------


def respond(
    backbone: Backbone,
    prompt: str,
    *,
    state: DiscourseState | None = None,
    compute: ComputeModule | None = None,
    confidence_floor: float = 0.5,
) -> Response:
    """Run the full BDRM pipeline on a prompt and return a response.

    Deterministic by default. When ``compute`` is supplied and the
    deterministic confidence falls below ``confidence_floor``, the
    Compute Module (§4) is invoked with a Conditioning payload built
    from the deterministic state. The CM may return a replacement
    string or ``None`` to defer to the deterministic answer.
    """
    if state is None:
        state = DiscourseState()

    # Step 1: prompt-to-backbone inference (M2).
    result = infer(backbone, prompt)

    # Step 2: pick primary anchor. If the prompt is a pronoun reference
    # ("What is it?", "Tell me about them") and the discourse state
    # has an active topic, use that topic as the anchor — this is the
    # multi-turn coreference path (BDRM §3.3).
    anchor_id = _primary_anchor(backbone, result)
    pronoun_resolved = False
    if (
        anchor_id is None
        and state.active_topic_node_ids
        and _prompt_has_pronoun(prompt)
    ):
        anchor_id = state.active_topic_node_ids[0]
        pronoun_resolved = True

    # Elaboration short-circuit: 'tell me more', 'continue', 'and?' —
    # the rule-based intent classifier sees these as instruction or
    # social, but the conversational meaning is "give me another fact
    # about the active topic". Route directly to the relation walker
    # when an active topic exists.
    elaboration = _is_elaboration(prompt)
    if elaboration and state.active_topic_node_ids:
        topic_id = state.active_topic_node_ids[0]
        text, conf = _elaborate_topic(backbone, topic_id, state.emitted_facts)
        # Use the topic itself as the anchor so downstream code (e.g.
        # the Compute Module conditioning) sees a meaningful subject.
        anchor_id = topic_id
    # Step 3 + 4: content selection + surface realization based on intent.
    elif result.intent == "question_answering":
        if anchor_id is None:
            text, conf = "I don't know what you're asking about.", 0.2
        else:
            text, conf = _answer_question(backbone, anchor_id)
    elif result.intent == "instruction":
        text, conf = _instruction_acknowledgment(backbone, anchor_id)
    elif result.intent == "social":
        text, conf = _social_response(prompt)
    elif result.intent == "inform":
        text, conf = _inform_acknowledgment(backbone, anchor_id)
    else:
        text, conf = "Could you rephrase that?", 0.3

    # Pronoun-resolved anchors are inherently more uncertain than direct
    # lemma lookups — the resolution is a heuristic. Discount confidence
    # slightly so the Compute Module floor catches edge cases.
    if pronoun_resolved:
        conf = max(0.0, conf - 0.1)

    # Step 5: discourse state update.
    state.update(result, anchor_id=anchor_id, backbone=backbone)

    # Confidence audit — flag if low. Compute Module (§4) fires here.
    needs_compute = conf < confidence_floor
    compute_used = False
    if needs_compute and compute is not None:
        conditioning = build_conditioning(
            backbone, result,
            primary_anchor_id=anchor_id,
            deterministic_answer=text,
            deterministic_confidence=conf,
        )
        override = compute.complete(conditioning)
        if override is not None:
            text = override
            compute_used = True

    return Response(
        text=text,
        inference=result,
        primary_anchor_id=anchor_id,
        confidence=conf,
        needs_compute_module=needs_compute,
        state=state,
        compute_module_used=compute_used,
        pronoun_resolved=pronoun_resolved,
    )
