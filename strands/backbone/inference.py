"""Prompt-to-backbone inference (BDRM spec §3.2).

Pipeline:
  1. Tokenization & lemma mapping  →  list of (surface, lemma, candidate_node_ids)
  2. Disambiguation                →  one chosen node per content token,
                                       via graph-coherence constraint
                                       satisfaction over backbone edges.
  3. Activation spreading          →  BFS with edge-weight × confidence
                                       × decay, hop-limited.
  4. Subgraph extraction           →  nodes above threshold + linking
                                       edges.
  5. Uncertainty tagging           →  nodes/edges below confidence floor
                                       or with compute_on_conflict.
  6. Intent classification         →  rule-based for M2; promotes to
                                       learned head later.

Pure deterministic: no neural computation in this module. Reads only
the backbone (mmap views) plus the stop-word list and a small token
regex. The Compute Module (§4) will plug in later for low-confidence
disambiguations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from strands.backbone.loader import Backbone
from strands.backbone.schema import Rel
from strands.lemmatizer import lemmatize
from strands.tokenizer import STOP_WORDS

# Reasonable defaults — exposed to callers via InferenceConfig.
_DEFAULT_HOP_LIMIT = 2
_DEFAULT_DECAY = 0.5
_DEFAULT_ACTIVATION_THRESHOLD = 0.05
_DEFAULT_CONFIDENCE_FLOOR = 0x6000  # ~37% — below this is "uncertain"

_TOK_RE = re.compile(r"[A-Za-z']+|[.!?]")
_SENTENCE_BREAK = frozenset({".", "!", "?"})


# --- Result types ---------------------------------------------------------


@dataclass(slots=True)
class TokenCandidate:
    """One token's surface form, lemma, and the candidate node IDs in
    the backbone that the lemma resolves to."""
    surface: str
    lemma: str
    candidate_node_ids: list[int]


@dataclass(slots=True)
class InferenceResult:
    prompt: str
    tokens: list[TokenCandidate]
    # token_idx → chosen node_id (only for content tokens that resolved)
    anchors: dict[int, int]
    # node_id → activation score in [0.0, 1.0]
    activations: dict[int, float]
    subgraph_node_ids: set[int]
    uncertain_node_ids: set[int]
    intent: str
    unknowns: list[str] = field(default_factory=list)

    @property
    def anchor_node_ids(self) -> list[int]:
        return list(self.anchors.values())


@dataclass(slots=True)
class InferenceConfig:
    hop_limit: int = _DEFAULT_HOP_LIMIT
    decay: float = _DEFAULT_DECAY
    activation_threshold: float = _DEFAULT_ACTIVATION_THRESHOLD
    confidence_floor: int = _DEFAULT_CONFIDENCE_FLOOR
    # When > 1 candidate, max graph-coherence comparisons (caps quadratic blow-up).
    max_candidates_per_token: int = 8


# --- Step 1: tokenize + candidate lookup ---------------------------------


def _tokenize_for_inference(prompt: str) -> list[tuple[str, str | None]]:
    """Return ordered (surface_token, sentence_terminator) pairs.
    Sentence terminators are returned as themselves; other punctuation
    is dropped at the regex level."""
    out: list[tuple[str, str | None]] = []
    for tok in _TOK_RE.findall(prompt):
        if tok in _SENTENCE_BREAK:
            if out:
                out[-1] = (out[-1][0], tok)
        else:
            out.append((tok.lower(), None))
    return out


def map_tokens_to_candidates(
    backbone: Backbone, prompt: str,
) -> tuple[list[TokenCandidate], list[str]]:
    """Step 1 — tokenize, lemmatize, look up candidate node IDs."""
    raw = _tokenize_for_inference(prompt)
    tokens: list[TokenCandidate] = []
    unknowns: list[str] = []

    for surface, _term in raw:
        if surface in STOP_WORDS:
            continue
        # Try direct lookup first (avoids false-lemma cases).
        candidates = backbone.nodes_for_lemma(surface)
        lemma = surface
        if not candidates:
            lemma = lemmatize(surface)
            candidates = backbone.nodes_for_lemma(lemma)
        if not candidates:
            unknowns.append(surface)
            continue
        tokens.append(TokenCandidate(
            surface=surface, lemma=lemma, candidate_node_ids=candidates,
        ))

    return tokens, unknowns


# --- Step 2: graph-coherence disambiguation ------------------------------


def _build_neighbor_set(backbone: Backbone, node_id: int) -> set[int]:
    """1-hop neighborhood of a node: target IDs of its outgoing edges."""
    edges = backbone.edges_of(node_id)
    if edges.size == 0:
        return set()
    return set(int(t) for t in edges["target_id"])


def _link_strength(
    backbone: Backbone,
    src: int,
    targets: set[int],
) -> float:
    """Highest weighted-by-confidence edge from ``src`` to any node in
    ``targets``. Normalized to [0, 1]."""
    edges = backbone.edges_of(src)
    if edges.size == 0:
        return 0.0
    mask = np.isin(edges["target_id"], np.fromiter(targets, dtype=np.uint32))
    if not mask.any():
        return 0.0
    sel = edges[mask]
    # weight × confidence, both u16 → max product = 0xFFFFFFFF
    scores = sel["weight"].astype(np.float64) * sel["confidence"].astype(np.float64)
    return float(scores.max() / (0xFFFF * 0xFFFF))


def disambiguate(
    backbone: Backbone,
    tokens: list[TokenCandidate],
    *,
    cfg: InferenceConfig | None = None,
) -> dict[int, int]:
    """Step 2 — pick one node per token by graph coherence.

    For each token i and candidate c, score(c) is the sum over every
    other token j of the strongest edge from c to any of token j's
    candidates. The candidate with the highest score wins. Single-
    candidate tokens are anchored deterministically.
    """
    cfg = cfg or InferenceConfig()
    chosen: dict[int, int] = {}

    # Trim runaway candidate counts per spec §3.6 (compute budget).
    capped: list[list[int]] = []
    for tc in tokens:
        capped.append(tc.candidate_node_ids[: cfg.max_candidates_per_token])

    for i, candidates_i in enumerate(capped):
        if not candidates_i:
            continue
        if len(candidates_i) == 1:
            chosen[i] = candidates_i[0]
            continue

        best_node: int | None = None
        best_score = -1.0
        for c in candidates_i:
            score = 0.0
            for j, candidates_j in enumerate(capped):
                if i == j or not candidates_j:
                    continue
                # Best link from c to any of token j's candidates.
                score += _link_strength(backbone, c, set(candidates_j))
            if score > best_score:
                best_score = score
                best_node = c
        # Tiebreaker on zero-score: prefer the lowest node_id, which by
        # WordNet convention is the most-frequent sense.
        if best_score == 0.0:
            best_node = min(candidates_i)
        chosen[i] = int(best_node)

    return chosen


# --- Step 3: activation spreading ---------------------------------------


def spread_activation(
    backbone: Backbone,
    anchor_node_ids: list[int],
    *,
    cfg: InferenceConfig | None = None,
) -> dict[int, float]:
    """Step 3 — BFS spreading activation along edges, weighted by
    edge weight × confidence and decayed per hop.

    Returns ``{node_id: activation_in_[0,1]}``.
    """
    cfg = cfg or InferenceConfig()
    activations: dict[int, float] = {nid: 1.0 for nid in anchor_node_ids}
    frontier = list(anchor_node_ids)

    for hop in range(cfg.hop_limit):
        decay_factor = cfg.decay ** (hop + 1)
        next_frontier: list[int] = []
        for node_id in frontier:
            edges = backbone.edges_of(node_id)
            if edges.size == 0:
                continue
            sources = activations.get(node_id, 0.0)
            for e in edges:
                target = int(e["target_id"])
                w = int(e["weight"]) / 0xFFFF
                conf = int(e["confidence"]) / 0xFFFF
                contribution = sources * w * conf * decay_factor
                if contribution < cfg.activation_threshold * 0.1:
                    continue
                prev = activations.get(target, 0.0)
                if contribution > prev:
                    activations[target] = contribution
                    next_frontier.append(target)
        frontier = next_frontier

    # Clamp to [0, 1] — weight × confidence × decay can theoretically
    # combine above 1 across multiple paths, but we want a normalized
    # score for downstream reasoning.
    return {nid: min(1.0, score) for nid, score in activations.items()}


# --- Step 4: subgraph extraction ----------------------------------------


def extract_subgraph(
    activations: dict[int, float],
    *,
    cfg: InferenceConfig | None = None,
) -> set[int]:
    cfg = cfg or InferenceConfig()
    return {
        nid for nid, score in activations.items()
        if score >= cfg.activation_threshold
    }


# --- Step 5: uncertainty tagging ----------------------------------------


def tag_uncertain(
    backbone: Backbone,
    subgraph: set[int],
    *,
    cfg: InferenceConfig | None = None,
) -> set[int]:
    """Mark subgraph nodes whose strongest outgoing edge has confidence
    below the floor, or that have ``compute_on_conflict`` set on any
    incident edge."""
    cfg = cfg or InferenceConfig()
    uncertain: set[int] = set()
    for nid in subgraph:
        edges = backbone.edges_of(nid)
        if edges.size == 0:
            continue
        if int(edges["confidence"].max()) < cfg.confidence_floor:
            uncertain.add(nid)
            continue
        if int(edges["compute_on_conflict"].max()) > 0:
            uncertain.add(nid)
    return uncertain


# --- Step 6: rule-based intent classifier (NN replacement comes later) ---


_QUESTION_TOKENS = frozenset({"what", "who", "where", "when", "why", "how",
                              "which", "whose", "whom"})
_IMPERATIVE_FIRST_WORDS = frozenset({"please", "could", "would", "can",
                                      "tell", "show", "explain", "make",
                                      "give", "find", "list", "describe",
                                      "open", "close", "set", "do"})
_SOCIAL_TOKENS = frozenset({"hi", "hello", "hey", "thanks", "thank",
                            "sorry", "bye", "goodbye"})


def classify_intent(prompt: str, anchors: dict[int, int]) -> str:
    lower = prompt.strip().lower()
    if not lower:
        return "empty"
    first_word = lower.split()[0]

    if lower.endswith("?"):
        return "question_answering"
    if any(t in _QUESTION_TOKENS for t in lower.split()):
        return "question_answering"
    if first_word in _SOCIAL_TOKENS:
        return "social"
    if first_word in _IMPERATIVE_FIRST_WORDS:
        return "instruction"
    if not anchors:
        return "unknown"
    return "inform"


# --- Top-level pipeline -------------------------------------------------


def infer(
    backbone: Backbone,
    prompt: str,
    *,
    cfg: InferenceConfig | None = None,
) -> InferenceResult:
    cfg = cfg or InferenceConfig()
    tokens, unknowns = map_tokens_to_candidates(backbone, prompt)
    anchors = disambiguate(backbone, tokens, cfg=cfg)
    activations = spread_activation(
        backbone, list(anchors.values()), cfg=cfg,
    )
    subgraph = extract_subgraph(activations, cfg=cfg)
    uncertain = tag_uncertain(backbone, subgraph, cfg=cfg)
    intent = classify_intent(prompt, anchors)

    return InferenceResult(
        prompt=prompt,
        tokens=tokens,
        anchors=anchors,
        activations=activations,
        subgraph_node_ids=subgraph,
        uncertain_node_ids=uncertain,
        intent=intent,
        unknowns=unknowns,
    )
