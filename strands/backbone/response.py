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

from dataclasses import dataclass, field

from strands.backbone.inference import InferenceResult, infer
from strands.backbone.loader import Backbone
from strands.backbone.schema import Rel


# --- Single-turn discourse state stub (M3) ------------------------------


@dataclass(slots=True)
class DiscourseState:
    """Minimal multi-turn state. Stored across calls if you keep the
    same instance; reset for fresh sessions."""
    active_topic_node_ids: list[int] = field(default_factory=list)
    entity_register: dict[str, int] = field(default_factory=dict)  # surface→node_id
    last_intent: str = ""
    turn_count: int = 0

    def update(self, result: InferenceResult, *, anchor_id: int | None) -> None:
        self.turn_count += 1
        self.last_intent = result.intent
        # Top 5 activated nodes become the rolling topic.
        top = sorted(
            result.activations.items(), key=lambda x: -x[1],
        )[:5]
        self.active_topic_node_ids = [n for n, _ in top]
        if anchor_id is not None:
            for tok in result.tokens:
                if (idx := next(
                    (i for i, t in enumerate(result.tokens)
                     if t.surface == tok.surface), None,
                )) is not None and idx in result.anchors:
                    self.entity_register[tok.surface] = result.anchors[idx]


# --- Response container -------------------------------------------------


@dataclass(slots=True)
class Response:
    text: str
    inference: InferenceResult
    primary_anchor_id: int | None
    confidence: float
    needs_compute_module: bool = False
    state: DiscourseState | None = None


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
    sentence; final fallback is just the lemmas list."""
    subject = _best_lemma(backbone, anchor_id) or "it"
    gloss = backbone.gloss_for(anchor_id)
    if gloss:
        # WordNet glosses are lowercase definitions, often without a
        # subject. Render as: "<subject> is <gloss>." capitalized.
        first = gloss[0].lower() + gloss[1:] if len(gloss) > 1 else gloss
        text = f"{subject.capitalize()} is {first}."
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
) -> Response:
    """Run the full BDRM pipeline on a prompt and return a response.

    Pure deterministic — no NN. Compute Module integration point (M6)
    will set ``needs_compute_module=True`` when it's wired in.
    """
    if state is None:
        state = DiscourseState()

    # Step 1: prompt-to-backbone inference (M2).
    result = infer(backbone, prompt)

    # Step 2: pick primary anchor.
    anchor_id = _primary_anchor(backbone, result)

    # Step 3 + 4: content selection + surface realization based on intent.
    if result.intent == "question_answering":
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

    # Step 5: discourse state update.
    state.update(result, anchor_id=anchor_id)

    # Confidence audit — flag if low.
    needs_compute = conf < 0.5

    return Response(
        text=text,
        inference=result,
        primary_anchor_id=anchor_id,
        confidence=conf,
        needs_compute_module=needs_compute,
        state=state,
    )
