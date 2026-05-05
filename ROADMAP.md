# BDRM Implementation Roadmap

Living document. Tracks the gap between `bdrm-spec.md` and what's shipped on `strand-v2-deterministic`. Updated as work progresses.

## Status legend

- **✓** built and tested
- **◐** partial (column exists / minimal implementation)
- **✗** missing
- **B*N*** = blocker — unblocks the slot/compute architecture
- **F*N*** = filler — closes a spec gap but isn't on the critical path

## Block 1 — slot/compute interchange (the critical path)

The architectural change from "binary template-or-NN replacement" to the spec's per-leaf interchange. These five items must ship in order.

### B1 — Phrasal template store (§2.4) `IN PROGRESS`
Surface-form tables per spec §2.4: lemma string + POS constraints + register markers + **phrasal templates** + collocation patterns. Templates are the deterministic slot fills in §3.5.1.

Sub-tasks:
- B1.0 `Template`, `SlotSpec`, `FilledSlot` data structures
- B1.1 `TemplateStore`: lookup by `(response_shape | relation_type | concept_id, register)` returns ranked candidates
- B1.2 Initial seed templates per response shape (definition, yesno-yes, yesno-no, elaboration, location, ability, composition, purpose, cause, effect, inform-ack, social) and per relation type
- B1.3 `render_with_fillers()` returning slot-tagged output (literal | lemma | compute-pending)
- B1.4 Refactor existing `_answer_by_relation` / `_answer_question` / `_inform_acknowledgment_with_belief` to read from the store

### B2 — Response Structure Tree (§3.4.3) `PENDING`
Deterministic answerer returns a tree of communicative acts with conceptual leaves, not a flat string.

Sub-tasks:
- B2.0 `ResponseStructure` / `CommunicativeAct` / `ConceptualLeaf` types
- B2.1 Per-intent planners that emit trees (replace string-returning answerers)
- B2.2 Confidence-audit walker that tags low-conf leaves for compute

### B3 — Per-leaf surface realization (§3.5) `PENDING`
Tree-walk with the four-way decision per leaf:
- High-conf phrasal template + matching context → emit template (zero compute)
- Single lemma, no alternatives → emit lemma with inflection
- No template, composition needed → assemble from primitives via assembly rules
- Flagged for compute → invoke compute module with local context + leaf spec

Sub-tasks:
- B3.0 `realize(structure, store, compute=None) -> str | TokenStream`
- B3.1 Inflection helpers (article, plural, agreement)
- B3.2 Composition primitives + assembly rules
- B3.3 Per-leaf compute invocation with ±20-token local context

### B4 — Compute Module local-window inputs (§4.3) `PENDING`
Compute module currently receives the full `Conditioning` of a whole-response substitution. Spec wants:
- ±20 tokens of already-generated output
- Discourse state vector
- Intervention spec (reference resolution / factual / logical / creative / ambiguity)
- Candidate set with confidences

Sub-tasks:
- B4.0 `LocalContext` + `InterventionSpec` + `CandidateSet` dataclasses (replace flat `Conditioning` for slot calls)
- B4.1 Updated `ComputeRequest` type with intervention category + candidates + window
- B4.2 Compute module signature change: `complete(ComputeRequest) -> str | None`
- B4.3 Streaming output protocol so realization can pause for compute, then resume

### B5 — Divergence-based training (§4.6) `PENDING`
Compute module currently trains by imitating the whole deterministic answer. Spec: train only on positions where deterministic output diverges from a teacher.

Sub-tasks:
- B5.0 Teacher integration (Anthropic API for bootstrap)
- B5.1 Divergence collector: drive prompts through both paths, record per-token disagreements at slot positions
- B5.2 Slot-level training data: `(local_context, intervention_spec, candidate_set, teacher_token)`
- B5.3 Updated `RecordingComputeModule` that captures slot-level training rows
- B5.4 Updated training loop that learns slot-level fills

### B6 — Discourse state completion (§3.3) `PENDING`
Currently: active topic vector ✓, entity register ✓, history ✓, beliefs ✓, speech act ✓.
Missing: commitment stack, subgoal stack, user model sketch.

Sub-tasks:
- B6.0 `CommitmentStack`: things the system has asserted; consulted for consistency
- B6.1 `SubgoalStack`: open dialog acts (e.g., partially answered question)
- B6.2 `UserModel`: inferred knowledge level / emotional state / goals
- B6.3 Update rules for each, called from `respond()`
- B6.4 Surface in `Conditioning` for the compute module

## Block 2 — fillers (close spec gaps off the critical path)

### F1 — Populate edge confidence / context_volatility / compute_on_conflict
Columns exist in the dtype but builder writes 0. WordNet edges should be near-1.0 confidence; ConceptNet edges should use the source's score. `compute_on_conflict` triggers when sources disagree.

### F2 — Wikidata ingest
Largest data gap. Entity properties, factual triples, temporal/spatial grounding, domain-specific relationships.

### F3 — FrameNet ingest
For `frame_id` population; semantic frames enable role-aware intent classification.

### F4 — PropBank ingest
Predicate-argument structure for verbs.

### F5 — Cross-source conflict resolution at compile time
Flag low-confidence and `compute_on_conflict` where sources disagree.

### F6 — Frame-aware intent classifier
Replace rule-based with frame structures + relationship patterns per §3.2.6.

### F7 — Explicit goal selection + relevance-scored content selection (§3.4.1, §3.4.2)
Today this is implicit in intent dispatch. Spec wants an explicit goal layer and relevance scoring against the user model.

### F8 — Syntactic smoothing pass (§3.5.3)
Lightweight rule-based agreement / function-word insertion after surface realization.

### F9 — Streaming token output
Currently returns full response strings. Spec target: token-by-token stream so per-leaf compute can interleave.

### F10 — Learn the ~50k model-layer parameters (Phase 2 training, §5.2)
The seven parameter sets in §5.1 (activation spreading weights, intent thresholds, content selection prefs, structure planning heuristics, assembly rules, compute thresholds, discourse state update rules). Currently all hand-coded.

### F11 — Reinforcement fine-tuning on human quality judgments (Phase 4, §5.2)
Deploy + collect preference signals; tune model-layer params (esp. compute thresholds) against human satisfaction.

## Ordering rationale

B1 → B2 → B3 → B4 → B5 are sequenced as the slot/compute interchange. B6 can run in parallel.

After Block 1 ships, the system has the spec's intended architecture. Block 2 then improves quality without architectural change.

Wikidata + FrameNet (F2, F3) are valuable but additive — they make the backbone richer without changing the runtime contracts.

The model-layer parameter learning (F10) is the eventual Phase 2 goal but requires Block 1 in place first; learning slot-fill thresholds against templates that don't exist yet is impossible.

## Current state snapshot (commit `3c21791`)

| Component | Spec coverage |
|---|---|
| Backbone — node/edge layout | ✓ |
| Backbone — relation taxonomy | ✓ |
| Backbone — WordNet + ConceptNet ingest | ✓ |
| Backbone — gloss table | ✓ (extension) |
| Backbone — surface form tables | ✗ |
| Backbone — Wikidata / FrameNet / PropBank | ✗ |
| Backbone — confidence / volatility / compute_on_conflict written | ✗ |
| Model layer — prompt-to-backbone inference | ✓ |
| Model layer — discourse state | ◐ (5 of 5 spec components partial; commitment/subgoal/user-model missing) |
| Model layer — response formulation | ◐ (string output, not tree) |
| Model layer — surface realization | ◐ (binary template-or-NN, not per-leaf) |
| Model layer — learned params | ✗ (all hand-coded) |
| Compute module — architecture | ✓ |
| Compute module — slot inputs | ✗ (whole-response only) |
| Compute module — divergence training | ✗ (imitation only) |
| Compute module — invocation triggers | ◐ (confidence only) |
| Inference pipeline — flow | ✓ |
| Inference pipeline — streaming | ✗ |
