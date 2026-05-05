"""Yes/no question handling (BDRM §3.4 enhancement).

Polar questions are a distinct question class from wh-questions: they
demand a binary verdict ('yes', 'no', or 'I don't know') backed by
evidence from the backbone. They are the ground truth for negation
in conversation — without them, the NN can't learn to refute false
assertions or confirm true ones.

Patterns covered:
  'Is/Are X a Y?'       → HYPERNYM closure: yes if Y is in the
                          ancestor chain of X, else no
  'Can X Y?' / 'Do(es) X Y?' → CAPABLE_OF / direct edge presence
  'Is X Y?' (adjective) → HAS_PROPERTY edge presence
  'Does X have Y?'      → HAS_A / MERONYM edge presence

Output format: ('yes' | 'no' | 'unsure', evidence_lemmas, confidence).
The verdict is paired with a one-sentence justification ('Yes, a cat
is a feline.') so the user sees the reasoning chain. Negative answers
include the actual category for grounding ('No, a cat is not a dog;
a cat is a feline.').
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from strands.backbone.loader import Backbone
from strands.backbone.schema import Rel


# Yes/no question shapes. Order matters — more specific patterns
# come first so 'Does X have Y?' matches before the generic
# 'Does X Y?'. Each pattern captures (subject, predicate); the relation
# to test is fixed per pattern.
_YN_PATTERNS: tuple[tuple[re.Pattern[str], Rel, str], ...] = (
    # 'Does X have Y?' — meronymic check (must precede generic Does).
    (re.compile(r"^do(?:es)? (?:a |an |the )?([a-z][a-z' ]*?) have (?:a |an |the )?([a-z][a-z' ]*?)\??$"),
     Rel.HAS_A, "has"),
    # 'Is/Are X a/an Y?' — hypernym closure check.
    (re.compile(r"^(?:is|are) (?:a |an |the )?([a-z][a-z' ]*?) (?:a|an) ([a-z][a-z' ]*?)\??$"),
     Rel.HYPERNYM, "hypernym"),
    # 'Can X Y?' — capability check.
    (re.compile(r"^can (?:a |an |the )?([a-z][a-z' ]*?) ([a-z][a-z' ]*?)\??$"),
     Rel.CAPABLE_OF, "capable"),
    # 'Does/Do X Y?' — capability check via verb.
    (re.compile(r"^do(?:es)? (?:a |an |the )?([a-z][a-z' ]*?) ([a-z][a-z' ]*?)\??$"),
     Rel.CAPABLE_OF, "capable"),
    # 'Is/Are X Y?' — adjective property check (only single-word target).
    (re.compile(r"^(?:is|are) (?:a |an |the )?([a-z][a-z' ]*?) ([a-z]+)\??$"),
     Rel.HAS_PROPERTY, "property"),
)


def _article(word: str) -> str:
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"


@dataclass(slots=True)
class YesNoAnswer:
    verdict: str            # 'yes' | 'no' | 'unsure'
    text: str               # rendered answer sentence
    confidence: float
    subject_surface: str
    target_surface: str
    relation: Rel
    evidence_node_id: int = -1


def is_yesno_question(prompt: str) -> bool:
    """Cheap predicate: does the prompt look like a yes/no question?"""
    s = prompt.strip().lower()
    if not s.endswith("?"):
        # Allow no-? but require an aux start.
        pass
    starts = ("is ", "are ", "was ", "were ", "do ", "does ", "did ",
              "can ", "could ", "will ", "would ", "should ", "has ",
              "have ", "had ")
    return any(s.startswith(start) for start in starts)


def _resolve(backbone: Backbone, surface: str) -> int:
    """Mirror beliefs._resolve — first noun/verb sense, with cheap
    depluralization fallback."""
    candidates = backbone.nodes_for_lemma(surface)
    if not candidates and surface.endswith("s"):
        candidates = backbone.nodes_for_lemma(surface[:-1])
    if not candidates:
        return -1
    # Prefer entity/event senses.
    for nid in candidates:
        type_byte = int(backbone.nodes[nid]["concept_type"])
        if type_byte & 0x03:  # ENTITY | EVENT
            return nid
    return candidates[0]


def _hypernym_chain(
    backbone: Backbone, node_id: int, max_depth: int = 6,
) -> list[int]:
    """Walk HYPERNYM edges up to max_depth, returning the ancestor
    node IDs in order. Cycle-safe."""
    chain: list[int] = []
    seen: set[int] = {node_id}
    current = node_id
    for _ in range(max_depth):
        edges = backbone.edges_with_relation(current, Rel.HYPERNYM)
        if edges.size == 0:
            break
        # Take the highest-weight hypernym to follow the canonical chain.
        best_idx = int(edges["weight"].argmax())
        next_id = int(edges[best_idx]["target_id"])
        if next_id in seen:
            break
        chain.append(next_id)
        seen.add(next_id)
        current = next_id
    return chain


def _is_in_hypernym_closure(
    backbone: Backbone, subject_id: int, target_id: int,
    *, max_depth: int = 4, branch: int = 3,
) -> tuple[bool, list[int]]:
    """Is target_id reachable from subject_id by following HYPERNYM
    edges?

    We use a width-limited BFS: at each step, follow the top-``branch``
    highest-weight hypernym edges, capped at ``max_depth`` total.
    Strict canonical-chain (top-1) misses cases where the dominant
    parent is wrong due to WordNet sense merges (e.g. 'Paris' merged
    with a plant genus). Full BFS picks up spurious deep bridges
    through merged-sense nodes (e.g. 'pointer' device + dog breed).
    Top-K with shallow depth balances both failure modes.
    """
    import numpy as np
    if subject_id == target_id:
        return True, [subject_id]
    seen: set[int] = {subject_id}
    queue: list[tuple[int, list[int]]] = [(subject_id, [])]
    for _ in range(max_depth):
        next_queue: list[tuple[int, list[int]]] = []
        for current, path in queue:
            edges = backbone.edges_with_relation(current, Rel.HYPERNYM)
            if edges.size == 0:
                continue
            order = np.argsort(-edges["weight"].astype(np.int64))
            for idx in order[:branch]:
                target = int(edges[int(idx)]["target_id"])
                if target == target_id:
                    return True, path + [target]
                if target in seen:
                    continue
                seen.add(target)
                next_queue.append((target, path + [target]))
        queue = next_queue
        if not queue:
            break
    return False, []


def _has_relation_target(
    backbone: Backbone, subject_id: int, rel: Rel,
    target_ids: set[int],
) -> bool:
    """Direct edge check: does subject have an edge of ``rel`` to any
    of ``target_ids``? Yes/no questions resolve targets to multiple
    sibling senses (e.g. all 20 senses of 'fly') so the match has to
    be set-based, not single-id."""
    if not target_ids:
        return False
    edges = backbone.edges_with_relation(subject_id, rel)
    if edges.size == 0:
        return False
    return any(int(e["target_id"]) in target_ids for e in edges)


def _best_lemma(backbone: Backbone, node_id: int) -> str:
    if node_id < 0:
        return ""
    lemmas = backbone.lemmas_for(node_id)
    if not lemmas:
        return ""
    single = [l for l in lemmas if " " not in l]
    return (sorted(single, key=len)[0] if single else lemmas[0])


def _all_subject_senses(
    backbone: Backbone, surface: str, *, top_k: int = 3,
) -> list[int]:
    """Up to ``top_k`` highest-edge-count senses of the lemma, with
    cheap depluralization fallback. Used for cross-sense yes/no
    answering — we want to consider the dominant sense plus a couple
    of alternates, but not the long tail (e.g. 'cat' shouldn't sweep
    in the caterpillar/construction-equipment sense and falsely
    affirm 'Is a cat a vehicle?').
    """
    nids = list(backbone.nodes_for_lemma(surface))
    if not nids and surface.endswith("s"):
        nids = list(backbone.nodes_for_lemma(surface[:-1]))
    # Sort by edge density, prefer ENTITY/EVENT typed senses.
    def rank(nid: int) -> tuple[int, int]:
        type_byte = int(backbone.nodes[nid]["concept_type"])
        type_priority = 1 if (type_byte & 0x03) else 0
        return (type_priority, int(backbone.nodes[nid]["relationship_count"]))
    nids.sort(key=rank, reverse=True)
    return nids[:top_k]


def _all_target_senses(
    backbone: Backbone, surface: str,
) -> set[int]:
    """Every node id for a target surface — used set-style so direct-
    edge checks match any sibling sense, not just the one _resolve
    picked. Includes cheap depluralization."""
    nids: set[int] = set(backbone.nodes_for_lemma(surface))
    if not nids and surface.endswith("s"):
        nids.update(backbone.nodes_for_lemma(surface[:-1]))
    # Verb -ing/-s/-ed normalization is too lossy for this layer;
    # leave it for later if the empty-edge rate is high.
    return nids


_YES_SHAPE_BY_REL: dict[Rel, str] = {
    Rel.CAPABLE_OF:   "yesno_yes_capable",
    Rel.HAS_PROPERTY: "yesno_yes_property",
    Rel.HAS_A:        "yesno_yes_has",
}
_POSITIVE_PHRASE_BY_REL: dict[Rel, str] = {
    Rel.CAPABLE_OF:   "{s} can {t}",
    Rel.HAS_PROPERTY: "{s} is {t}",
    Rel.HAS_A:        "{s} has {t}",
}


def _render(shape: str, fillers: dict, *, fallback: str) -> str:
    from strands.realization import build_default_store, render
    # Local import to avoid a cycle at module load.
    global _STORE
    if "_STORE" not in globals() or _STORE is None:
        _STORE = build_default_store()
    template = _STORE.best(shape=shape)
    if template is None:
        return fallback
    return render(template, fillers).finalize()


_STORE = None


def answer_yesno(
    backbone: Backbone, prompt: str,
) -> YesNoAnswer | None:
    """Match the prompt against yes/no patterns and produce a verdict
    backed by backbone evidence. All surface text is rendered via the
    realization layer's template store (B1)."""
    from strands.realization import FillerValue
    s = prompt.strip().lower()
    for pattern, rel, kind in _YN_PATTERNS:
        m = pattern.match(s)
        if m is None:
            continue
        subject = m.group(1).strip()
        target = m.group(2).strip()
        if not subject or not target:
            continue
        subject_senses = _all_subject_senses(backbone, subject)
        target_senses = _all_target_senses(backbone, target)
        if not subject_senses or not target_senses:
            missing = subject if not subject_senses else target
            text = _render(
                "definition_unknown",
                {"subject": FillerValue(missing)},
                fallback=f"I don't have enough information about {missing}.",
            )
            return YesNoAnswer(
                verdict="unsure", text=text, confidence=0.3,
                subject_surface=subject, target_surface=target,
                relation=rel,
            )
        target_id = max(
            target_senses,
            key=lambda nid: int(backbone.nodes[nid]["relationship_count"]),
        )

        if kind == "hypernym":
            for sid in subject_senses:
                yes, _ = _is_in_hypernym_closure(backbone, sid, target_id)
                if yes:
                    text = _render(
                        "yesno_yes_hypernym",
                        {
                            "subject": FillerValue(subject),
                            "target": FillerValue(target),
                        },
                        fallback=f"Yes, {_article(subject)} {subject} is "
                                 f"{_article(target)} {target}.",
                    )
                    return YesNoAnswer(
                        verdict="yes", text=text, confidence=0.92,
                        subject_surface=subject, target_surface=target,
                        relation=rel, evidence_node_id=target_id,
                    )
            best_sid = max(
                subject_senses,
                key=lambda nid: (
                    bool(int(backbone.nodes[nid]["concept_type"]) & 0x01),
                    int(backbone.nodes[nid]["relationship_count"]),
                ),
            )
            chain = _hypernym_chain(backbone, best_sid, max_depth=3)
            if chain:
                actual = _best_lemma(backbone, chain[0])
                if actual and actual != target:
                    text = _render(
                        "yesno_no_with_actual",
                        {
                            "subject": FillerValue(subject),
                            "target": FillerValue(target),
                            "actual": FillerValue(actual),
                        },
                        fallback=(
                            f"No, {_article(subject)} {subject} is not "
                            f"{_article(target)} {target}; "
                            f"{_article(subject)} {subject} is "
                            f"{_article(actual)} {actual}."
                        ),
                    )
                    return YesNoAnswer(
                        verdict="no", text=text, confidence=0.88,
                        subject_surface=subject, target_surface=target,
                        relation=rel, evidence_node_id=chain[0],
                    )
            text = _render(
                "yesno_no_plain",
                {
                    "subject": FillerValue(subject),
                    "target": FillerValue(target),
                },
                fallback=f"No, {_article(subject)} {subject} is not "
                         f"{_article(target)} {target}.",
            )
            return YesNoAnswer(
                verdict="no", text=text, confidence=0.8,
                subject_surface=subject, target_surface=target,
                relation=rel,
            )

        # capable / property / has paths.
        yes_sid: int | None = None
        for sid in subject_senses:
            if _has_relation_target(backbone, sid, rel, target_senses):
                yes_sid = sid
                break
        if yes_sid is not None:
            shape = _YES_SHAPE_BY_REL.get(rel, "yesno_yes_property")
            text = _render(
                shape,
                {
                    "subject": FillerValue(subject),
                    "target": FillerValue(target),
                },
                fallback=f"Yes, {_article(subject)} {subject} is {target}.",
            )
            return YesNoAnswer(
                verdict="yes", text=text, confidence=0.85,
                subject_surface=subject, target_surface=target,
                relation=rel, evidence_node_id=target_id,
            )

        # Unsure path: render the positive phrase via a tiny inline
        # template, then plug it into the yesno_unsure shape's literal
        # slot. Avoids double-negation.
        positive_phrase = _POSITIVE_PHRASE_BY_REL.get(
            rel, "{s} is {t}",
        ).format(s=subject, t=target)
        text = _render(
            "yesno_unsure",
            {"positive_phrase": FillerValue(
                positive_phrase, source="literal",
            )},
            fallback=(
                "I don't have direct evidence either way about whether "
                f"{positive_phrase}."
            ),
        )
        return YesNoAnswer(
            verdict="unsure", text=text, confidence=0.4,
            subject_surface=subject, target_surface=target,
            relation=rel,
        )
    return None
