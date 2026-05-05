# Backbone-Driven Reasoning Model (BDRM) — Full Architecture Specification

## 1. System Overview

BDRM is a hybrid AI architecture that separates stable world knowledge from dynamic reasoning. It consists of three core components operating in a coordinated pipeline:

| Component | Hardware | Function |
|-----------|----------|----------|
| **Semantic Backbone** | CPU + RAM | Compressed, pre-compiled knowledge graph storing concepts, relationships, and linguistic renderings |
| **Model Layer** | CPU (primarily) | Reasoning engine that navigates the backbone, formulates response structures, and manages discourse state |
| **Compute Module** | GPU (small, sparse use) | High-precision neural resolver invoked only when backbone + model layer cannot resolve a decision confidently |

The architecture's defining property: **the model thinks in concepts, not tokens. Language is output, not cognition.**

---

## 2. The Semantic Backbone

### 2.1 Purpose

A persistent, versioned, pre-compiled knowledge resource that provides:
- Conceptual grounding for all system operations
- Deterministic, zero-compute retrieval of facts, relationships, and linguistic patterns
- The primary workspace for model layer reasoning

### 2.2 Source Data

Initial compilation draws from:

| Source | Contribution |
|--------|--------------|
| **WordNet** | Lexical-semantic relationships (synonymy, hypernymy, meronymy, antonymy), synset structure, lemma groupings |
| **ConceptNet** | Commonsense relationships (causes, usedFor, hasProperty, capableOf, desires, atLocation), weighted assertions |
| **Wikidata** | Entity properties, factual triples, temporal and spatial grounding, domain-specific relationships |
| **FrameNet** (supplementary) | Semantic frames and role structures for verb-centered event patterns |
| **PropBank** (supplementary) | Predicate-argument structures for verb usage patterns |

### 2.3 Internal Data Model

The backbone is a directed, typed, weighted multigraph.

**Node Structure (fixed-width: 128 bytes per node)**

| Field | Bytes | Description |
|-------|-------|-------------|
| `node_id` | 4 | Unique integer identifier |
| `concept_type` | 1 | Bit field encoding: entity, event, property, relation, frame, abstract, quantifier |
| `activation_default` | 2 | Base activation weight (0-65535) |
| `volatility_flag` | 1 | How mutable this concept's relationships are under context (0=definitional, 255=highly contextual) |
| `embedding_compressed` | 16 | Locality-sensitive hash of semantic neighborhood for fast approximate matching |
| `lemma_count` | 1 | Number of surface forms linked |
| `lemma_offset` | 4 | Pointer to lemma table |
| `relationship_count` | 1 | Number of outgoing edges |
| `relationship_offset` | 4 | Pointer to edge table |
| `frame_id` | 2 | Link to FrameNet frame if applicable |
| `language_independent_id` | 4 | Cross-lingual concept identifier |
| `reserved` | 88 | Expansion space for domain-specific annotations |

**Edge Structure (fixed-width: 32 bytes per edge)**

| Field | Bytes | Description |
|-------|-------|-------------|
| `target_id` | 4 | Destination node |
| `relation_type` | 2 | Encoded relation type from a fixed taxonomy |
| `weight` | 2 | Association strength (0-65535) |
| `confidence` | 2 | Epistemic confidence: how certain this edge is across contexts (0=speculative, 65535=definitional) |
| `context_volatility` | 2 | How much this edge's applicability shifts with discourse context |
| `source_attribution` | 1 | Which source database(s) asserted this edge |
| `bidirectional_flag` | 1 | Whether the inverse holds |
| `compute_on_conflict` | 1 | Flag: if this edge is contradicted by context, must invoke compute module |
| `reserved` | 17 | |

**Relation Type Taxonomy (16-bit, extensible)**

Core types encoded:
- `HYPERNYM` / `HYPONYM`
- `MERONYM` / `HOLONYM`
- `SYNONYM` / `ANTONYM`
- `CAUSES` / `CAUSED_BY`
- `HAS_PROPERTY` / `PROPERTY_OF`
- `CAPABLE_OF`
- `AT_LOCATION` / `LOCATION_OF`
- `PART_OF`
- `USED_FOR`
- `HAS_PREREQUISITE`
- `ENTAILS`
- `TEMPORAL_BEFORE` / `TEMPORAL_AFTER`
- `COREFERENTIAL`
- `DERIVED_FROM`
- `CONTEXTUALLY_ASSOCIATED` (catch-all for statistical co-occurrence)

### 2.4 Surface Form Tables

Stored separately but linked from nodes. Variable-width entries containing:
- Lemma string (UTF-8, null-terminated)
- Part of speech constraints
- Register markers (formal, informal, technical, archaic)
- Phrasal templates where this concept appears as a constituent
- Collocation patterns with adjacent concept types

### 2.5 Compression Strategy

Total uncompressed size estimate for 500,000 nodes, 5 million edges:
- Nodes: 500,000 × 128 = 64 MB
- Edges: 5,000,000 × 32 = 160 MB
- Surface forms: variable, estimated ~50-80 MB
- **Total: approximately 300 MB uncompressed**

Further compression achievable via:
- Delta encoding of edge offsets
- Huffman coding on high-frequency relation types
- Shared lemmas across nodes (many concepts share surface forms)

Target: **under 200 MB** in RAM for the full backbone. This is the size of a single image file.

### 2.6 Compilation Process

One-time build from source databases:

1. **Ingestion:** Parse WordNet, ConceptNet, Wikidata into a unified intermediate representation
2. **Entity resolution:** Merge co-referring entities across sources using shared lemma matches and Wikidata cross-references
3. **Conflict resolution:** Where sources disagree on a relationship, flag with low confidence and mark `compute_on_conflict`
4. **Weight normalization:** Rescale source-specific weights to the unified 0-65535 range
5. **Graph optimization:** Topological sort for cache-friendly traversal ordering, remove redundant transitive edges where the compressed path is sufficient
6. **Serialization:** Pack into the binary format above

### 2.7 Updating

The backbone is versioned. Updates to source databases trigger incremental rebuilds that preserve node IDs where possible. The model layer receives a notification of version changes and can adapt its trained weights accordingly.

---

## 3. The Model Layer

### 3.1 Purpose

The model layer is the reasoning core. It:
- Processes user prompts against the backbone
- Maintains discourse state across conversation turns
- Formulates response structures before surface rendering
- Decides when to invoke the compute module
- Manages the surface realization of response structures into natural language

### 3.2 Prompt-to-Backbone Inference (Step 1)

Given a user prompt:

1. **Tokenization and lemma mapping:** Map surface tokens to backbone node candidates using the surface form tables. Multiple candidates per token are retained.

2. **Disambiguation pass:** For each ambiguous token, evaluate which candidate's relationship neighborhood best fits the other activated concepts in the prompt. This is a constraint satisfaction walk, not a neural computation. The backbone's edge weights provide the constraints.

3. **Conceptual activation spreading:** From the disambiguated anchor nodes, spread activation along high-weight, high-confidence edges up to a configurable hop limit (default: 2 hops). Activated nodes receive an activation score based on edge weights and distance.

4. **Subgraph extraction:** The activated nodes and their interconnecting edges form the **prompt subgraph** — a bounded, relevant subset of the backbone that the model layer will reason over.

5. **Uncertainty tagging:** Nodes and edges with confidence below a threshold, or with `compute_on_conflict` flags triggered, are tagged for potential compute module attention.

6. **Intent classification:** Using the activated frame structures and relationship patterns, classify the user's likely communicative intent (question-answering, explanation-request, instruction, social, creative, etc.). This intent tag shapes response formulation.

### 3.3 Discourse State Management

The model layer maintains a running discourse state across turns:

| State Component | Description |
|-----------------|-------------|
| **Active topic vector** | Which regions of the backbone are currently salient |
| **Entity register** | Referents introduced and their recency/salience scores |
| **Commitment stack** | Things the system has asserted and is now committed to |
| **Conversational subgoal stack** | Active dialog acts not yet completed (e.g., partially answered question) |
| **User model sketch** | Inferred user knowledge level, emotional state, goals |

This state is updated after each user prompt and after each system response. It fits in kilobytes and is CPU-maintained.

### 3.4 Response Formulation (Step 2)

This is the central reasoning act. Given the prompt subgraph and discourse state:

1. **Goal selection:** Based on intent classification and discourse state, select the communicative goal(s) for the response (inform, explain, question, acknowledge, redirect, etc.).

2. **Content selection:** Traverse the prompt subgraph to identify the conceptual material relevant to the goal. Apply relevance scoring that accounts for user model, discourse coherence, and informativeness.

3. **Structure planning:** Assemble the selected content into a response structure — an ordered tree of communicative acts with conceptual content at the leaves. The structure respects discourse coherence principles (given-before-new, topic continuity, logical ordering for explanations).

4. **Confidence audit:** For each conceptual element in the response structure, check backbone confidence. If any element falls below threshold, mark it for compute module intervention during surface realization.

### 3.5 Surface Realization (Step 3)

The response structure is rendered into natural language tokens:

1. **Traverse the response structure tree** in output order.

2. **For each conceptual leaf:**
   - If the backbone has a high-confidence phrasal template matching the concept and the required syntactic context → retrieve and use it (deterministic, zero compute).
   - If the concept maps to a single lemma with no contextual alternatives → emit the lemma with appropriate inflection.
   - If the concept requires composition (multiple words, no template) → assemble from primitives using learned assembly rules.
   - If the concept is flagged for compute intervention → invoke the compute module with the local context and concept specification.

3. **Syntactic smoothing:** After all leaves are realized, apply a lightweight syntactic pass that handles agreement, ordering adjustments, and function word insertion. This is rule-based with learned exceptions.

The surface realization step is the only point where actual English tokens are generated. The entire reasoning process happens in the conceptual space.

---

## 4. The Compute Module

### 4.1 Purpose

A small neural model invoked sparsely to handle cases the backbone and model layer cannot resolve deterministically.

### 4.2 Architecture

A compact transformer or state-space model, approximately 200-500 million parameters. Small enough for microsecond-range inference on a consumer GPU.

### 4.3 Inputs

When invoked, it receives:
- **Local context window:** ±20 tokens of already-generated output
- **Discourse state vector:** Compressed representation from the model layer
- **Intervention specification:** What kind of decision is needed (reference resolution, factual verification, logical inference, creative generation, ambiguity resolution)
- **Candidate set:** The top candidates from the backbone's deterministic lookup, with confidence scores

### 4.4 Outputs

- **Resolved token(s):** The correct filler for the slot, or a short sequence
- **Updated confidence:** For this intervention, how confident was the resolution (feeds back into training)
- **Discourse state delta:** Any updates to the discourse state based on what was resolved

### 4.5 Invocation Triggers

The compute module is invoked when:
- The backbone's confidence on a node or edge falls below the learned threshold
- The `compute_on_conflict` flag is set and a context conflict is detected
- The model layer's assembly rules cannot produce a coherent rendering
- The discourse state contains an unresolved reference that the deterministic traversal can't lock onto

### 4.6 Training

The compute module is trained on **divergence cases** — moments where the deterministic system's best output differs from the desired output. Training data is generated by:
1. Running a full-scale teacher model on diverse prompts
2. Running the BDRM system on the same prompts
3. Collecting all positions where BDRM's deterministic output diverges from the teacher
4. Each divergence becomes a training example for the compute module: given the context at that position and the deterministic candidate, predict the teacher's token

The compute module is periodically retrained as the backbone is updated.

---

## 5. Training The Model Layer

### 5.1 What Is Learned

The model layer has relatively few learned parameters compared to conventional models:

| Parameter Set | Approximate Size | What It Controls |
|---------------|------------------|------------------|
| Activation spreading weights | ~5,000 parameters | Which relationship types are prioritized during prompt-to-backbone inference |
| Intent classification thresholds | ~500 parameters | How activation patterns map to communicative intent categories |
| Content selection preferences | ~10,000 parameters | How relevance is scored given discourse state and user model |
| Structure planning heuristics | ~5,000 parameters | Ordering and organization of response structures |
| Assembly rules for surface realization | ~20,000 parameters | How conceptual structures map to syntactic patterns |
| Compute invocation thresholds | ~1,000 parameters | When to trigger the compute module |
| Discourse state update rules | ~5,000 parameters | How to update state after user input and system output |

**Total: approximately 50,000 learned parameters.** This is the size of a small statistical model, not a neural network. Training is fast and data-efficient.

### 5.2 Training Regimen

**Phase 1: Backbone compilation (no model layer involved)**
Build the backbone from source databases as described in 2.6.

**Phase 2: Initial model layer training**
Using a corpus of human conversational data, train the model layer to:
- Correctly map prompts to backbone subgraphs
- Produce response structures that match human response patterns
- Render responses fluently

Training signal: Supervised learning on human conversation data, with the backbone providing the conceptual substrate. For each prompt-response pair, the expected backbone activation and response structure are derived by running the prompt through the backbone and aligning the response to the backbone's conceptual space.

**Phase 3: Divergence collection and compute module training**
Run the trained system against a diverse prompt set. Collect divergence cases. Train the compute module on those cases. Iterate.

**Phase 4: Reinforcement fine-tuning**
Deploy the system in interactive settings. Collect human quality judgments on responses. Use those judgments to fine-tune the model layer's parameters — particularly the compute invocation thresholds and content selection preferences. The reward signal is human satisfaction. The backbone remains frozen.

**Phase 5: Continuous adaptation**
During deployment, track when the model layer's confidence diverges from actual output quality. Use these signals to continuously adjust:
- Backbone edge confidence scores (slowly, with human-in-the-loop validation)
- Compute invocation thresholds (more rapidly, automated)
- Assembly rule preferences (in response to user feedback)

### 5.3 Why Training Is Efficient

The backbone provides the knowledge. Training only tunes behavior. The model layer isn't learning that Paris is the capital of France — that's in the backbone. It's learning how to have a coherent conversation about capitals, countries, and geopolitics using the knowledge it already has access to.

New domains require backbone updates (new Wikidata ingest), not full retraining. The model layer's reasoning patterns transfer across domains because they operate on the conceptual level.

---

## 6. Inference Pipeline

### 6.1 End-to-End Flow (Per User Turn)

```
User Prompt
    │
    ▼
[Prompt-to-Backbone Inference]  ← CPU, deterministic
    │  Maps tokens → concepts
    │  Disambiguates, activates subgraph
    │  Tags uncertainties
    │
    ▼
[Discourse State Integration]  ← CPU, model layer weights
    │  Merges prompt subgraph with existing state
    │  Updates entity register, topic vector
    │  Classifies intent
    │
    ▼
[Response Formulation]  ← CPU, model layer weights
    │  Selects content from subgraph
    │  Plans response structure tree
    │  Audits confidence at each node
    │
    ▼
[Surface Realization]  ← CPU, model layer + occasional GPU
    │  Traverses structure tree
    │  Retrieves templates / emits tokens / invokes compute
    │  Applies syntactic smoothing
    │
    ▼
[Output Tokens Streamed to User]
    │
    ▼
[Discourse State Updated]  ← CPU
    │  Adds system commitments
    │  Adjusts salience weights
    │  Prepares for next turn
```

### 6.2 Performance Characteristics (Target)

| Metric | Target |
|--------|--------|
| Prompt-to-backbone latency | <5ms |
| Response formulation latency | <20ms |
| Surface realization (per token, deterministic) | <0.1ms |
| Compute module invocation (per intervention) | <5ms |
| Typical interventions per response | 2-5 |
| Typical total latency per turn | 100-500ms |
| RAM usage (backbone + model state) | <500 MB |
| GPU usage (idle except during interventions) | <1 GB VRAM |

This is a system that could run on a laptop with integrated graphics, responding faster than human typing speed.

---

## 7. Advantages Over Current Architectures

| Dimension | Conventional LLM | BDRM |
|-----------|-----------------|------|
| Knowledge storage | Distributed across billions of weights | Explicit, inspectable, correctable |
| Reasoning medium | Token prediction | Conceptual manipulation |
| Compute per token | Constant, high | Variable, mostly near-zero |
| Hardware requirement | GPU for all inference | CPU for 90-95% of operations |
| Factual updates | Require retraining or RAG | Backbone update, no retraining |
| Hallucination source | Probabilistic generation | Confined to compute module interventions |
| Explainability | Opaque | Traceable through backbone traversal |
| Training data required | Trillions of tokens | Backbone from curated sources + conversation data for behavior |
| Long-context coherence | Attention over full window | Persistent discourse state + conceptual tracking |

---

## 8. Known Challenges & Open Problems

### 8.1 Backbone Coverage

WordNet, ConceptNet, and Wikidata provide broad coverage but are not exhaustive. Gaps in commonsense knowledge, domain-specific terminology, or cultural knowledge will result in missing backbone nodes. The system needs a mechanism for recognizing when a concept is absent and either:
- Invoking the compute module to handle it purely neurally
- Flagging it for human-in-the-loop backbone expansion

### 8.2 Backbone Quality

Source databases contain errors, contradictions, and culturally biased assertions. The compilation step must handle conflicts gracefully. The `compute_on_conflict` flag is a start, but systemic biases could be baked into the backbone and require auditing.

### 8.3 The Expressiveness Gap

Can a graph-based backbone with learned assembly rules produce the full range of natural, creative, contextually appropriate language? This is the central empirical question. Humans reason conceptually and render linguistically, so the architecture is plausible. But the backbone must be rich enough to support the full expressive range. This likely requires iteration.

### 8.4 Compute Module Boundary

The division of labor between deterministic and compute paths is learned, not hand-designed. Getting the thresholds right — so that the compute module is invoked when needed but not otherwise — will require careful training and possibly per-domain tuning.

### 8.5 Cold Start

Before deployment, the model layer has no discourse experience. Phase 2 training requires quality conversational data aligned to backbone concepts. This data may need to be generated initially using a conventional LLM as a teacher, which introduces a dependency on existing technology for bootstrapping.

---

## Summary

BDRM is a cognitive architecture that:

- Stores world knowledge in a compressed, explicit, CPU-addressable backbone built from curated sources
- Reasons in a conceptual space rather than a token space
- Generates language as a rendering step, not a cognitive step
- Invokes neural computation only for the small fraction of decisions that can't be resolved deterministically
- Maintains discourse coherence through a persistent state rather than recomputing from a full context window
- Learns behavior from modest amounts of conversational data rather than trillions of tokens

The result is a system that could achieve near-LLM conversational quality with a fraction of the compute, running primarily on CPU and RAM, with factual knowledge that is inspectable, correctable, and updatable without retraining.
