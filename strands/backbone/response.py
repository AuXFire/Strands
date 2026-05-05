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
from strands.backbone.speech_act import classify_speech_act
from strands.backbone.yesno import answer_yesno, is_yesno_question
from strands.realization import (
    ActType,
    CommunicativeAct,
    ConceptualLeaf,
    FillerValue,
    RealizedResponse,
    ResponseStructure,
    TemplateStore,
    build_default_store,
    realize,
    render,
    sequence,
    single_leaf,
)


# Module-level template store, lazy-initialized on first use so the
# realization layer can be inspected/swapped per test.
_TEMPLATE_STORE: TemplateStore | None = None


def _store() -> TemplateStore:
    global _TEMPLATE_STORE
    if _TEMPLATE_STORE is None:
        _TEMPLATE_STORE = build_default_store()
    return _TEMPLATE_STORE


def set_template_store(store: TemplateStore | None) -> None:
    """Override the module-level template store (tests / swapping)."""
    global _TEMPLATE_STORE
    _TEMPLATE_STORE = store


def _render_shape(
    shape: str, fillers: dict, *, fallback: str = "",
    confidence: float = 0.7,
) -> tuple[str, float]:
    """Look up the best template for ``shape`` and render it. When no
    template matches, return ``fallback`` (caller's last-resort string)
    or an empty string."""
    template = _store().best(shape=shape)
    if template is None:
        return fallback, confidence
    rendered = render(template, fillers)
    return rendered.finalize(), rendered.confidence


def _render_relation(
    rel: Rel, fillers: dict, *, shape: str = "",
    fallback: str = "", confidence: float = 0.85,
) -> tuple[str, float]:
    """Look up the best template for the (shape, relation) pair and
    render it. Falls back to a relation-only lookup if no shape match."""
    template = (
        _store().best(shape=shape) if shape else None
    ) or _store().best(relation_type=int(rel))
    if template is None:
        return fallback, confidence
    rendered = render(template, fillers)
    return rendered.finalize(), rendered.confidence
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


# Question-type → (relation, response_shape).
# The response_shape selects the template; the relation drives the
# backbone walk. Templates live in strands/realization/seeds.py.
_QUESTION_TYPE_RELATIONS: dict[str, tuple[Rel, str]] = {
    "location":    (Rel.AT_LOCATION,  "answer_location"),
    "composition": (Rel.MADE_OF,      "answer_composition"),
    "purpose":     (Rel.USED_FOR,     "answer_purpose"),
    "ability":     (Rel.CAPABLE_OF,   "answer_ability"),
    "cause":       (Rel.CAUSED_BY,    "answer_cause"),
    "effect":      (Rel.CAUSES,       "answer_effect"),
}


# Relation walk priority order for topic elaboration. Each relation
# resolves to a template via TemplateStore.by_relation(rel).
_ELABORATION_RELATIONS: tuple[Rel, ...] = (
    Rel.HYPERNYM,
    Rel.HAS_PROPERTY,
    Rel.MERONYM,
    Rel.AT_LOCATION,
    Rel.USED_FOR,
    Rel.CAPABLE_OF,
    Rel.MADE_OF,
    Rel.HYPONYM,
    Rel.HOLONYM,
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


def _plan_answer_by_relation(
    backbone: Backbone, anchor_id: int, rel: Rel,
    *, top_k: int = 2, subject_override: str | None = None,
    shape: str = "",
) -> ResponseStructure | None:
    """Walk a specific relation from the anchor and produce a single-
    leaf ASSERT structure pointing at the matching template. Returns
    None if no edges exist or no template matches."""
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
    # Resolve which template the leaf should target. Prefer the
    # response_shape; fall back to relation-only.
    if shape and _store().best(shape=shape) is not None:
        template_shape = shape
        template_relation = 0
    elif _store().best(relation_type=int(rel)) is not None:
        template_shape = ""
        template_relation = int(rel)
    else:
        return None
    return single_leaf(
        ActType.ASSERT,
        template_shape=template_shape,
        template_relation=template_relation,
        fillers={
            "subject": FillerValue(subject),
            "target": targets,  # LEMMA_LIST handles the join
        },
        confidence=0.85,
        intent="question_answering",
        backbone_node_ids=(anchor_id,),
    )


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


def _plan_question(
    backbone: Backbone, anchor_id: int,
    *, question_type: str = "definition",
    user_surface: str | None = None,
) -> ResponseStructure:
    """Plan a response structure for a wh-question. Definition (default)
    falls back to the gloss; typed questions walk a specific relation
    and produce an ASSERT structure citing the matching template."""
    subject = (
        user_surface or _best_lemma(backbone, anchor_id) or "it"
    )

    # Typed questions try the specific relation first.
    if question_type != "definition":
        rel_shape = _QUESTION_TYPE_RELATIONS.get(question_type)
        if rel_shape is not None:
            rel, shape = rel_shape
            structure = _plan_answer_by_relation(
                backbone, anchor_id, rel, shape=shape,
                subject_override=subject,
            )
            if structure is not None:
                return structure
            # WSD may have picked a sense without this relation. Try
            # a sibling sense of the same lemma.
            sibling = _sibling_sense_with_relation(backbone, anchor_id, rel)
            if sibling is not None:
                structure = _plan_answer_by_relation(
                    backbone, sibling, rel, shape=shape,
                    subject_override=subject,
                )
                if structure is not None:
                    # Slightly lower confidence — we crossed a sense.
                    new_conf = max(0.6, structure.confidence - 0.1)
                    structure.confidence = new_conf
                    structure.root.confidence = new_conf
                    if structure.root.leaf is not None:
                        structure.root.leaf.confidence = new_conf
                    return structure

    # Definition path: gloss-backed.
    gloss = backbone.gloss_for(anchor_id)
    if gloss:
        if len(gloss) < 40:
            hyp = _hypernym_target(backbone, anchor_id)
            if hyp:
                target_lemma = _best_lemma(backbone, hyp[0])
                if target_lemma and target_lemma != subject:
                    return single_leaf(
                        ActType.ASSERT,
                        template_shape="definition_with_hypernym",
                        fillers={
                            "subject": FillerValue(subject),
                            "gloss": FillerValue(gloss),
                            "hypernym": FillerValue(target_lemma),
                        },
                        confidence=0.9,
                        intent="question_answering",
                        backbone_node_ids=(anchor_id, hyp[0]),
                    )
        return single_leaf(
            ActType.ASSERT,
            template_shape="definition",
            fillers={
                "subject": FillerValue(subject),
                "gloss": FillerValue(gloss),
            },
            confidence=0.9,
            intent="question_answering",
            backbone_node_ids=(anchor_id,),
        )

    hyp = _hypernym_target(backbone, anchor_id)
    if hyp:
        target_id, weight = hyp
        target_lemma = _best_lemma(backbone, target_id) or "something"
        confidence = min(0.85, weight / 0xFFFF + 0.4)
        return single_leaf(
            ActType.ASSERT,
            template_shape="definition_fallback",
            fillers={
                "subject": FillerValue(subject),
                "hypernym": FillerValue(target_lemma),
            },
            confidence=confidence,
            intent="question_answering",
            backbone_node_ids=(anchor_id, target_id),
        )

    # Nothing useful — DEFER act flagged for compute would normally
    # land here, but we keep emitting the deterministic 'definition_
    # unknown' template to preserve current behavior. B3 will switch
    # to a compute-flagged leaf when the compute module is wired in.
    return single_leaf(
        ActType.DEFER,
        template_shape="definition_unknown",
        fillers={"subject": FillerValue(subject)},
        confidence=0.2,
        intent="question_answering",
        backbone_node_ids=(anchor_id,),
    )


def _plan_elaborate(
    backbone: Backbone, anchor_id: int,
    emitted: set[tuple[int, int, int]],
) -> ResponseStructure:
    """Plan an ELABORATE structure: walk relations from the anchor in
    priority order; the first highest-weight target not in ``emitted``
    becomes a single-leaf structure tagged for the matching relation
    template. When all relations are exhausted, returns an
    ``elaborate_exhausted`` leaf at low confidence."""
    subject = _best_lemma(backbone, anchor_id) or "it"
    for rel in _ELABORATION_RELATIONS:
        edges = backbone.edges_with_relation(anchor_id, rel)
        if edges.size == 0:
            continue
        # Confirm a template exists for this (shape, relation) pair.
        if (
            _store().best(shape="elaborate", relation_type=int(rel))
            is None
            and _store().best(relation_type=int(rel)) is None
        ):
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
            return single_leaf(
                ActType.ELABORATE,
                template_shape="elaborate",
                template_relation=int(rel),
                fillers={
                    "subject": FillerValue(subject),
                    "target": FillerValue(target_lemma),
                },
                confidence=0.8,
                intent="question_answering",
                backbone_node_ids=(anchor_id, target_id),
            )
    return single_leaf(
        ActType.ELABORATE,
        template_shape="elaborate_exhausted",
        fillers={"subject": FillerValue(subject)},
        confidence=0.4,
        intent="question_answering",
        backbone_node_ids=(anchor_id,),
    )


def _plan_inform_ack(
    backbone: Backbone, anchor_id: int | None,
) -> ResponseStructure:
    if anchor_id is None:
        return single_leaf(
            ActType.ACKNOWLEDGE,
            template_shape="inform_bare",
            confidence=0.6,
            intent="inform",
        )
    subject = _best_lemma(backbone, anchor_id) or "that"
    return single_leaf(
        ActType.ACKNOWLEDGE,
        template_shape="inform_generic_ack",
        fillers={"subject": FillerValue(subject)},
        confidence=0.7,
        intent="inform",
        backbone_node_ids=(anchor_id,),
    )


def _plan_belief_ack(belief: Belief) -> ResponseStructure:
    """When inform-turn belief extraction succeeds, the acknowledgement
    echoes the user's own words rather than a re-templated form — this
    avoids subject-verb agreement bugs ('cats is cute') and confirms
    the exact wording we registered."""
    raw = belief.raw_prompt.rstrip(".!?")
    return single_leaf(
        ActType.ACKNOWLEDGE,
        template_shape="inform_belief_ack",
        fillers={"raw": FillerValue(raw, source="literal")},
        confidence=0.85,
        intent="inform",
    )


def _plan_instruction_ack(
    backbone: Backbone, anchor_id: int | None,
) -> ResponseStructure:
    if anchor_id is None:
        return single_leaf(
            ActType.INSTRUCT_RESPONSE,
            template_shape="instruct_bare",
            confidence=0.5,
            intent="instruction",
        )
    subject = _best_lemma(backbone, anchor_id) or "the request"
    return single_leaf(
        ActType.INSTRUCT_RESPONSE,
        template_shape="instruct_with_subject",
        fillers={"subject": FillerValue(subject)},
        confidence=0.6,
        intent="instruction",
        backbone_node_ids=(anchor_id,),
    )


_SOCIAL_SHAPES: tuple[tuple[tuple[str, ...], str, ActType, float], ...] = (
    (("hi", "hello", "hey"), "social_greeting", ActType.GREET, 0.95),
    (("thank",), "social_thanks", ActType.THANK_RESPONSE, 0.95),
    (("sorry",), "social_apology", ActType.APOLOGIZE_RESPONSE, 0.9),
    (("bye",), "social_farewell", ActType.FAREWELL, 0.95),
)


def _plan_social(prompt: str) -> ResponseStructure:
    lower = prompt.strip().lower()
    for cues, shape, act, conf in _SOCIAL_SHAPES:
        if any(c in lower for c in cues):
            return single_leaf(
                act, template_shape=shape, confidence=conf, intent="social",
            )
    return single_leaf(
        ActType.GREET,
        template_shape="social_fallback",
        confidence=0.7,
        intent="social",
    )


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
    structure: ResponseStructure | None = None

    # Elaboration short-circuit: 'tell me more', 'continue', 'and?' —
    # the rule-based intent classifier sees these as instruction or
    # social, but the conversational meaning is "give me another fact
    # about the active topic". Route directly to the relation walker
    # when an active topic exists.
    elaboration = _is_elaboration(prompt)
    if elaboration and state.active_topic_node_ids:
        topic_id = state.active_topic_node_ids[0]
        structure = _plan_elaborate(backbone, topic_id, state.emitted_facts)
        # Use the topic itself as the anchor so downstream code (e.g.
        # the Compute Module conditioning) sees a meaningful subject.
        anchor_id = topic_id
    # Step 3 + 4: content selection + structure planning by intent.
    elif result.intent == "question_answering":
        # Yes/no questions are a distinct class — try them first.
        yn = answer_yesno(backbone, prompt) if is_yesno_question(prompt) else None
        if yn is not None:
            qtype = "yesno"
            # YesNoAnswer.text is already the rendered string. Wrap
            # it as a single-leaf VERIFY structure with a literal-only
            # leaf so the realize() step is a no-op pass-through.
            structure = single_leaf(
                ActType.VERIFY,
                template_id="",            # no template
                fillers={},
                confidence=yn.confidence,
                intent="question_answering",
            )
            # Stash the prerendered text on the leaf via a literal
            # filler against an inline template-less leaf. We honor it
            # by short-circuiting realize() below when text is preset.
            structure.root.leaf.fillers["__yn_text"] = FillerValue(
                yn.text, source="literal", confidence=yn.confidence,
            )
        elif anchor_id is None:
            structure = single_leaf(
                ActType.DEFER,
                template_shape="fallback_no_anchor",
                confidence=0.2,
                intent="question_answering",
            )
        else:
            qtype = _classify_question_type(prompt)
            user_surface = next(
                (
                    t.surface for i, t in enumerate(result.tokens)
                    if result.anchors.get(i) == anchor_id
                ),
                None,
            )
            structure = _plan_question(
                backbone, anchor_id,
                question_type=qtype, user_surface=user_surface,
            )
    elif result.intent == "instruction":
        structure = _plan_instruction_ack(backbone, anchor_id)
    elif result.intent == "social":
        structure = _plan_social(prompt)
    elif result.intent == "inform":
        belief = extract_belief(
            backbone, prompt, turn_index=state.turn_count + 1,
        )
        if belief is not None:
            state.session_beliefs.append(belief)
            structure = _plan_belief_ack(belief)
        else:
            structure = _plan_inform_ack(backbone, anchor_id)
    else:
        structure = single_leaf(
            ActType.REPHRASE_REQUEST,
            template_shape="fallback_rephrase",
            confidence=0.3,
        )

    # Realize the structure to surface text. yesno results carry their
    # prerendered text on a __yn_text filler; honor that as a special
    # case until B3 lifts yesno into the planner directly.
    assert structure is not None
    if (
        structure.root.is_leaf
        and structure.root.leaf is not None
        and "__yn_text" in structure.root.leaf.fillers
    ):
        yn_filler = structure.root.leaf.fillers["__yn_text"]
        text = yn_filler.value if isinstance(yn_filler, FillerValue) else ""
        conf = structure.confidence
    else:
        realized = realize(structure, _store())
        text, conf = realized.text, structure.confidence

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
        speech_act = classify_speech_act(
            prompt, intent=result.intent, question_type=qtype,
        )
        conditioning = build_conditioning(
            backbone, result,
            primary_anchor_id=anchor_id,
            deterministic_answer=text,
            deterministic_confidence=conf,
            history=state.history,
            user_beliefs=state.session_beliefs,
            speech_act=speech_act,
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
