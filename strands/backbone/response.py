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

from strands.backbone.beliefs import Belief, extract_belief
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


# Question-type classification. Each pattern routes the question to a
# relation walker that pulls a more specific answer than the default
# gloss-based definition.
_QUESTION_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("location",    ("where ",)),
    ("composition", ("made of", "made out of", "consist of", "what's in",
                     "what is in")),
    ("purpose",     ("used for", "what is .* for", " purpose of")),
    ("ability",     ("what does", "what do", "what can", "how does",
                     "how do", "able to", "capable of")),
    ("cause",       ("what causes", "why does", "why do",
                     "what makes")),
    ("effect",      ("what does .* cause", "what happens because")),
)


def _classify_question_type(prompt: str) -> str:
    """Detect the wh-pattern in a question and return one of:
    'definition' (default, gloss-based) | 'location' | 'composition'
    | 'purpose' | 'ability' | 'cause' | 'effect'."""
    s = prompt.strip().lower().rstrip(".?!")
    for qtype, patterns in _QUESTION_TYPES:
        for pat in patterns:
            if pat.startswith("what is .*") or pat.startswith("what does .*"):
                # Tiny regex-ish: prefix and suffix must both occur,
                # in that order. Avoids importing re again per call.
                head, _, tail = pat.partition(".*")
                head, tail = head.strip(), tail.strip()
                hi = s.find(head)
                if hi != -1 and tail and s.find(tail, hi + len(head)) != -1:
                    return qtype
            elif pat in s:
                return qtype
    return "definition"


# Question-type → (relation, sentence template).
_QUESTION_TYPE_RELATIONS: dict[str, tuple[Rel, str]] = {
    "location":    (Rel.AT_LOCATION,  "{subject} can be found at {target}"),
    "composition": (Rel.MADE_OF,      "{subject} is made of {target}"),
    "purpose":     (Rel.USED_FOR,     "{subject} is used for {target}"),
    "ability":     (Rel.CAPABLE_OF,   "{subject} is capable of {target}"),
    "cause":       (Rel.CAUSED_BY,    "{subject} is caused by {target}"),
    "effect":      (Rel.CAUSES,       "{subject} causes {target}"),
}


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
class TurnRecord:
    """One past turn, frozen for replay into the Compute Module
    conditioning. Keeps just enough to reconstruct the conversation
    without dragging the whole InferenceResult forward."""
    turn_index: int
    prompt: str
    response: str
    intent: str
    question_type: str = ""
    primary_anchor_id: int | None = None
    primary_anchor_lemma: str = ""
    pronoun_resolved: bool = False
    confidence: float = 0.0


@dataclass(slots=True)
class DiscourseState:
    """Multi-turn state. Stored across calls if you keep the same
    instance; reset for fresh sessions."""
    active_topic_node_ids: list[int] = field(default_factory=list)
    entity_register: dict[str, int] = field(default_factory=dict)  # surface→node_id
    last_intent: str = ""
    turn_count: int = 0
    # Facts emitted by the elaboration walker so successive
    # 'tell me more' calls produce novel content. Key:
    # (subject_node_id, relation_id, target_node_id).
    emitted_facts: set[tuple[int, int, int]] = field(default_factory=set)
    # Rolling history, oldest-first, capped at history_limit. Each
    # TurnRecord is what the Compute Module sees as the conversation
    # so far when generating a response.
    history: list[TurnRecord] = field(default_factory=list)
    history_limit: int = 16
    # User-stated beliefs lifted from inform turns (BDRM §2.3 volatility).
    # These are session-scoped: held in memory, not persisted to the
    # backbone, but visible to the Compute Module via Conditioning.
    session_beliefs: list[Belief] = field(default_factory=list)

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

    def append_turn(self, record: TurnRecord) -> None:
        """Append a turn record and trim the history to history_limit."""
        self.history.append(record)
        if len(self.history) > self.history_limit:
            # Keep the most recent N — older turns drop off.
            del self.history[: len(self.history) - self.history_limit]


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


def _answer_by_relation(
    backbone: Backbone, anchor_id: int, rel: Rel, template: str,
    *, top_k: int = 2, subject_override: str | None = None,
) -> tuple[str, float] | None:
    """Walk a specific relation from the anchor and return up to
    ``top_k`` highest-weight targets joined into a single sentence,
    or ``None`` if no edges exist. ``subject_override`` lets the caller
    keep the user's surface word ('house') rather than the matched
    node's primary lemma ('home')."""
    edges = backbone.edges_with_relation(anchor_id, rel)
    if edges.size == 0:
        return None
    subject = subject_override or _best_lemma(backbone, anchor_id) or "it"
    order = np.argsort(-edges["weight"].astype(np.int64))
    targets: list[str] = []
    for idx in order:
        target_id = int(edges[int(idx)]["target_id"])
        target_lemma = _best_lemma(backbone, target_id)
        if not target_lemma or target_lemma == subject:
            continue
        if target_lemma in targets:
            continue
        targets.append(target_lemma)
        if len(targets) >= top_k:
            break
    if not targets:
        return None
    if len(targets) == 1:
        joined = targets[0]
    else:
        joined = " and ".join(targets)
    sentence = template.format(subject=subject, target=joined)
    return sentence[0].upper() + sentence[1:] + ".", 0.85


def _sibling_sense_with_relation(
    backbone: Backbone, anchor_id: int, rel: Rel,
) -> int | None:
    """When the chosen anchor has no edges of ``rel``, the WSD may
    have picked the wrong sense for a typed question. Search every
    sense that shares ANY lemma with the anchor and return the one
    with the most edges of the requested relation, if any."""
    lemmas = backbone.lemmas_for(anchor_id)
    if not lemmas:
        return None
    seen: set[int] = {anchor_id}
    best: tuple[int, int] | None = None  # (count, node_id)
    for lemma in lemmas:
        for nid in backbone.nodes_for_lemma(lemma):
            if nid in seen:
                continue
            seen.add(nid)
            edges = backbone.edges_with_relation(nid, rel)
            if edges.size == 0:
                continue
            count = int(edges.size)
            if best is None or count > best[0]:
                best = (count, nid)
    return best[1] if best is not None else None


def _answer_question(
    backbone: Backbone, anchor_id: int,
    *, question_type: str = "definition",
    user_surface: str | None = None,
) -> tuple[str, float]:
    """Route a question to a relation-specific answerer based on the
    detected wh-pattern. Definition (default) falls back to the gloss;
    typed questions (where/what does X do/etc.) walk a specific edge
    type and return that answer when available, else fall back to the
    definition.

    ``user_surface`` is the word the user actually typed. When supplied
    and singular-canonical, it's used as the subject so we don't render
    'Home is made of wood' for someone who asked about 'house'."""
    subject = (
        user_surface or _best_lemma(backbone, anchor_id) or "it"
    )

    # Typed questions try the specific relation first.
    if question_type != "definition":
        rel_template = _QUESTION_TYPE_RELATIONS.get(question_type)
        if rel_template is not None:
            rel, template = rel_template
            answer = _answer_by_relation(
                backbone, anchor_id, rel, template,
                subject_override=subject,
            )
            if answer is not None:
                return answer
            # WSD may have picked a sense without this relation. Try
            # a sibling sense of the same lemma — typed questions are
            # more confident about WHAT they're asking than which sense.
            sibling = _sibling_sense_with_relation(backbone, anchor_id, rel)
            if sibling is not None:
                answer = _answer_by_relation(
                    backbone, sibling, rel, template,
                    subject_override=subject,
                )
                if answer is not None:
                    # Slightly lower confidence — we crossed a sense.
                    return answer[0], max(0.6, answer[1] - 0.1)

    # Definition path (and fallback for typed questions): use gloss.
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


def _inform_acknowledgment_with_belief(
    belief: Belief,
) -> tuple[str, float]:
    """When inform-turn belief extraction succeeds, the acknowledgement
    echoes the user's own words rather than a re-templated form — this
    avoids subject-verb agreement bugs ('cats is cute') and confirms
    the exact wording we registered. Higher confidence than the generic
    'Got it' since we actually structured the fact."""
    return f"Got it — noted that {belief.raw_prompt.rstrip('.!?')}.", 0.85


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

    # Track question_type for both the answer call and the TurnRecord.
    qtype = ""

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
            qtype = _classify_question_type(prompt)
            # Find the surface the user typed for the chosen anchor,
            # so the answer reads with their word ('house') rather than
            # the node's primary lemma ('home').
            user_surface = next(
                (
                    t.surface for i, t in enumerate(result.tokens)
                    if result.anchors.get(i) == anchor_id
                ),
                None,
            )
            text, conf = _answer_question(
                backbone, anchor_id,
                question_type=qtype, user_surface=user_surface,
            )
    elif result.intent == "instruction":
        text, conf = _instruction_acknowledgment(backbone, anchor_id)
    elif result.intent == "social":
        text, conf = _social_response(prompt)
    elif result.intent == "inform":
        # Try to lift the declarative into a structured Belief that
        # we keep on the discourse state. The acknowledgement template
        # echoes the (subject, relation, target) when extraction worked
        # so the user knows we registered the fact.
        belief = extract_belief(
            backbone, prompt, turn_index=state.turn_count + 1,
        )
        if belief is not None:
            state.session_beliefs.append(belief)
            text, conf = _inform_acknowledgment_with_belief(belief)
        else:
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
    # The conditioning carries discourse history (excluding this turn)
    # so the NN sees the conversation context.
    needs_compute = conf < confidence_floor
    compute_used = False
    if needs_compute and compute is not None:
        conditioning = build_conditioning(
            backbone, result,
            primary_anchor_id=anchor_id,
            deterministic_answer=text,
            deterministic_confidence=conf,
            history=state.history,
            user_beliefs=state.session_beliefs,
        )
        override = compute.complete(conditioning)
        if override is not None:
            text = override
            compute_used = True

    # Append this turn to history AFTER the Compute Module has run so the
    # next turn's conditioning includes the final response.
    anchor_lemma = (
        _best_lemma(backbone, anchor_id) if anchor_id is not None else ""
    )
    state.append_turn(TurnRecord(
        turn_index=state.turn_count,
        prompt=prompt,
        response=text,
        intent=result.intent,
        question_type=qtype,
        primary_anchor_id=anchor_id,
        primary_anchor_lemma=anchor_lemma,
        pronoun_resolved=pronoun_resolved,
        confidence=conf,
    ))

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
