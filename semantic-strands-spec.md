# Semantic Strands — Technical Specification v1.0

## A Deterministic, Interpretable Replacement for Vector Embeddings

---

## 1. Executive Summary

Semantic Strands encodes meaning as short, structured byte sequences instead of high-dimensional float vectors. Every word, code token, or data element maps to a **codon** (3-byte semantic address) plus a **shade byte** (nuance modifier), producing a 4-byte encoding per token — a **1,536x reduction** versus standard embeddings (e.g., OpenAI `text-embedding-3-small` at 6,144 bytes/token).

Strands are compared via sequence alignment (borrowed from bioinformatics), not cosine similarity. The system is fully deterministic, requires no GPU inference, runs on any hardware, and produces human-readable output where you can inspect *exactly why* two inputs matched.

### What This Replaces

| Capability | Embeddings | Semantic Strands |
|---|---|---|
| Storage per token | 6,144 bytes (1536×f32) | 4 bytes (codon+shade) |
| Comparison | Cosine similarity (opaque) | Sequence alignment (interpretable) |
| Infrastructure | Vector DB (Pinecone, Qdrant, pgvector) | Any DB — SQL, SQLite, flat files, grep |
| Inference | Model call per encode | Hash map lookup — O(1) |
| Hardware | GPU recommended | Raspberry Pi |
| Vendor lock-in | Tied to embedding model | None — static codebook |
| Interpretability | None — 1536 opaque floats | Full — every codon is readable |
| Re-indexing on model change | Full re-embed required | Never — codebook is stable |

---

## 2. Architecture Overview

```
INPUT (text, code, or structured data)
  │
  ▼
┌─────────────┐
│  TOKENIZER  │  Language-aware tokenization
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LEMMATIZER │  Normalize inflections → base forms
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  CODEBOOK   │  Static hash map: token → codon
│  LOOKUP     │  ~500K entries, ~50MB, versioned
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SHADE     │  Compute shade byte from context
│  DERIVATION │  [intensity|abstraction|formality|polarity]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   STRAND    │  Ordered sequence of codon:shade pairs
│  ASSEMBLY   │  Output: "EM001:5D·AC402:71·NA312:44"
└─────────────┘
```

### Pipeline Components

1. **Tokenizer** — Splits input into semantic units. For natural language: words. For code: AST-aware tokens. For structured data: field-value pairs.
2. **Lemmatizer** — Reduces inflected forms to base form ("running"→"run", "better"→"good"). Uses language-specific rules + exception dictionary.
3. **Codebook** — The core data structure. A static hash map from lemmatized tokens to semantic addresses (codons). Built from WordNet, Datamuse, ConceptNet, and language-specific corpora. Versioned and human-editable.
4. **Shade Derivation** — Computes a uint8 shade byte packing 4 orthogonal dimensions into 2 bits each. Uses lexicon lookups + contextual rules.
5. **Strand Assembly** — Concatenates codons with shades into an ordered sequence. This is the final output — analogous to an embedding vector but readable and tiny.

---

## 3. The Codon — Semantic Address Format

Each codon is a **3-byte structured address** encoding domain, category, and concept:

```
Byte 0: Domain (uint8)     — Top-level semantic field (Emotion, Action, Object, etc.)
Byte 1: Category (uint8)   — Sub-field within domain (e.g., Emotion→Attachment)
Byte 2: Concept (uint8)    — Specific meaning cluster (e.g., Attachment→Love)
```

### String Representation

For readability, codons are rendered as: `{Domain}{Category}{Concept}` — e.g., `EM106` = Emotion, category 1 (Attachment), concept 06 (Love).

### Comparison Scoring

Codons are compared hierarchically, with partial credit at each level:

```
Same domain only:                    0.25  (25%)
Same domain + same category:         0.40  (40%)
Same domain + category + concept:    0.75  (75%)
Exact match with shade proximity:    0.75 + (shade_similarity × 0.25) → up to 1.0
```

This means "happy" and "sad" (same domain: Emotion, different concept) score ~0.25, while "happy" and "joyful" (same concept) score ~0.95+. "happy" and "database" score 0.

---

## 4. The Shade Byte — Nuance Encoding

A single uint8 packing 4 dimensions into 2 bits each:

```
Bit layout: [II AA FF PP]

  II (bits 7-6): Intensity
    00 = whisper    (slightly, somewhat, mildly)
    01 = normal     (standard intensity)
    10 = strong     (very, highly, deeply)
    11 = extreme    (extremely, overwhelmingly, devastatingly)

  AA (bits 5-4): Abstraction
    00 = concrete   (short, physical words: "dog", "run", "hot")
    01 = moderate   (medium-length words: "happy", "forest")
    10 = abstract   (longer words: "democracy", "philosophy")
    11 = ethereal   (very abstract: "transcendence", "epistemological")

  FF (bits 3-2): Formality
    00 = casual     (slang, informal: "dude", "gonna", "awesome")
    01 = neutral    (standard register)
    10 = formal     (professional: "commence", "facilitate")
    11 = academic   (scholarly: "epistemology", "hermeneutics")

  PP (bits 1-0): Polarity
    00 = negative   (words with negative sentiment)
    01 = neutral-   (slightly neutral, no strong sentiment)
    10 = neutral+   (slightly positive or truly neutral)
    11 = positive   (words with positive sentiment)
```

### Shade Computation

```python
def compute_shade(token: str, context: list[str]) -> int:
    polarity = sentiment_lexicon.get(token, 1)    # 0-3
    abstraction = min(3, len(token) // 4)          # heuristic: word length
    formality = formality_lexicon.get(token, 1)    # 0-3
    intensity = intensity_lexicon.get(token, 1)    # 0-3, can be boosted by adverbs in context

    return (intensity << 6) | (abstraction << 4) | (formality << 2) | polarity
```

### Context-Aware Shade Modification

When encoding sentences (not isolated words), the shade of a word can be modified by its context:

- **Intensity adverbs** boost the intensity bits: "very happy" → "happy" gets intensity=2 instead of 1
- **Negation** flips polarity: "not happy" → "happy" gets polarity=0 instead of 3
- **Formal context markers** boost formality: "the committee hereby resolves" → all words get formality boost

---

## 5. Domain Taxonomy

### 5.1 Natural Language Domains

The following domains cover general natural language. Each domain has 8-16 categories with 16-64 concepts per category.

```
DOMAIN   CODE  CATEGORIES (examples)
──────── ────  ─────────────────────
Emotion   EM   basic(happy/sad/angry/afraid/calm), attachment(love/hate),
               surprise, confidence, energy, social, complex
Action    AC   locomotion, consumption, verbal, mental, creative,
               manipulation, perception, search, connect/separate, combat, work
Object    OB   buildings, vehicles, electronics, reading, tools, structural,
               furniture, valuables, clothing, containers
Quality   QU   size, speed, goodness, beauty, age, temperature, difficulty,
               strength, light, cleanliness, wealth, truth, importance,
               safety, interest, intelligence, texture, weight
Abstract  AB   mental(ideas/plans/problems), truth/fiction, power/freedom,
               achievement, existence, knowledge, chance/chaos
Nature    NA   animals(mammals/birds/fish/reptiles/insects), plants(trees/flowers),
               geography(water/weather/terrain), celestial(sun/moon/stars/elements)
Person    PE   general, family, social, occupations(medical/education/military/
               arts/athletics/legal/commercial)
Space     SP   places(cities/nations), positions(inside/outside/near/far/direction)
Time      TM   temporal(present/past/future), durations(moments/hours/years),
               frequency(always/never/often/rarely), rate(sudden/gradual)
Quantity  QT   totality(all/none/some), comparison(more/less/enough)
Body      BD   head/face, arms/hands, torso, legs/feet, organs, tissues, fluids
Social    SO   groups(family/community/teams), governance(government/politics/war),
               institutions(schools/hospitals), commerce(markets/business)
Tech      TC   software(code/algorithms/databases), web(internet/networking),
               AI/security, data/hardware
Food      FD   grains, produce, dairy, beverages, sweets, meals/dining, cooking
Comm.     CM   language/text, messages/media, meetings/events
Sensory   SN   colors(warm/cool/neutral), textures, tastes, sounds
Movement  MV   rotation, vibration, flow
Relation  RL   togetherness, causation, similarity, difference
Economy   EC   systems, trade, pricing, investment
```

### 5.2 Code Domains

These domains handle source code semantics. They operate alongside text domains — code comments and string literals route through text domains while code structure routes through code domains.

```
DOMAIN         CODE  CATEGORIES
─────────────  ────  ─────────────────────
Control Flow   CF    sequential, conditional(if/switch/ternary), loop(for/while/
                     do/foreach/map), recursion, branching(break/continue/return/
                     throw/goto), async(await/promise/callback/future/channel),
                     concurrency(thread/process/lock/mutex/semaphore/atomic)

Data Structure DS    primitive(int/float/string/bool/char/byte), linear(array/list/
                     vector/deque/buffer/stack/queue), associative(map/dict/set/
                     hashmap/hashtable), tree(binary/bst/avl/btree/trie/heap),
                     graph(directed/undirected/dag/weighted), composite(struct/
                     record/tuple/union/enum)

Type System    TS    declaration(var/let/const/type/interface/class/trait/protocol),
                     annotation(generic/template/nullable/optional/readonly/mutable),
                     conversion(cast/coerce/parse/serialize/deserialize/marshal),
                     constraint(extends/implements/where/bound)

Operation      OP    arithmetic(add/sub/mul/div/mod/pow/abs/round), comparison(eq/
                     neq/lt/gt/lte/gte), logical(and/or/not/xor), bitwise(shift/
                     mask/toggle), string(concat/split/trim/replace/match/format/
                     interpolate), collection(map/filter/reduce/sort/find/group/
                     flatten/zip/slice/splice/push/pop/insert/delete/merge)

IO             IO    file(read/write/append/delete/move/copy/watch/stream), network(
                     http/websocket/grpc/rest/graphql/fetch/request/response),
                     database(query/insert/update/delete/transaction/migrate/index),
                     console(log/debug/warn/error/print/input/prompt), environment(
                     env/config/arg/flag/secret)

Error          ER    handling(try/catch/finally/throw/raise/panic/recover), validation(
                     assert/check/verify/sanitize/validate/guard), result(ok/err/
                     some/none/success/failure/maybe/either), logging(log/trace/debug/
                     info/warn/error/fatal)

Pattern        PT    creational(factory/builder/singleton/prototype/pool), structural(
                     adapter/bridge/composite/decorator/facade/proxy/flyweight),
                     behavioral(observer/strategy/command/iterator/mediator/state/
                     visitor/chain), architectural(mvc/mvvm/repository/middleware/
                     pipeline/plugin/event-driven/pub-sub/cqrs)

Module         MD    import/export/require/module/package/crate/namespace/scope/
                     closure/dependency/injection/registry/resolution

Testing        TE    unit(test/assert/expect/mock/stub/spy/fake/fixture), integration(
                     setup/teardown/suite/describe/it/before/after), coverage(line/
                     branch/statement/function), benchmark(bench/perf/profile/measure)

API            AP    endpoint(route/path/handler/controller/middleware), auth(token/
                     session/cookie/oauth/jwt/api-key), schema(openapi/graphql/protobuf/
                     json-schema/avro), rate(limit/throttle/retry/backoff/circuit-breaker)

Infrastructure IN    container(docker/k8s/pod/service/ingress), ci-cd(build/test/
                     deploy/release/rollback), cloud(compute/storage/queue/cache/cdn/
                     serverless/lambda/function), monitoring(metric/log/trace/alert/
                     dashboard/healthcheck/sla)
```

### 5.3 Structured Data Domains (optional, for data pipeline use)

```
DOMAIN         CODE  CATEGORIES
─────────────  ────  ─────────────────────
Schema         SC    table/column/field/key/index/constraint/relation/view/migration
Transform      TR    filter/map/reduce/join/group/pivot/aggregate/window/normalize
Format         FM    json/xml/csv/yaml/toml/parquet/avro/protobuf/msgpack
```

---

## 6. Codebook Construction

The codebook is the central data structure. It must be built from real linguistic and code data — not generated by AI.

### 6.1 Natural Language Codebook Sources

Build the codebook by merging data from these sources, in priority order:

```
SOURCE              WHAT IT PROVIDES                        PRIORITY
────────────────    ──────────────────────────────────      ────────
WordNet 3.1         Synsets, hypernyms, hyponyms,           1 (primary)
                    meronyms. ~155K words, ~117K synsets.
                    Use synset hierarchy to derive
                    domain→category→concept mapping.

ConceptNet 5        Relational knowledge graph. Use          2
                    /r/IsA, /r/RelatedTo, /r/Synonym
                    edges to expand concept groups and
                    cross-validate WordNet mappings.

Datamuse API        Real-time synonym/related-word           3
                    lookups. Use ml= (meaning like)
                    and rel_syn= (synonyms) endpoints
                    to fill gaps. Free, CORS-enabled,
                    no auth required.

English Wiktionary  Definitions, etymology, usage notes.     4
                    Use for formality/register classification
                    and domain disambiguation.

SentiWordNet 3.0    Sentiment scores per synset.             shade
                    Direct mapping to polarity bits
                    in shade byte.

word_frequency      Corpus frequency data. Use to            filter
(wordfreq package)  prioritize common words and filter
                    out extremely rare entries.
```

### 6.2 Code Codebook Sources

```
SOURCE                  WHAT IT PROVIDES
──────────────────      ──────────────────────────────────
tree-sitter grammars    AST node types for 100+ languages.
                        Map node types to code domain codons.

Language specs          Reserved keywords, built-in functions,
(MDN, Python docs,      standard library names. Each keyword
Rust reference, etc.)   maps to a specific codon.

GitHub code corpus      Identifier frequency data. Prioritize
(via BigQuery or        common function/variable naming patterns.
GH Archive)             Use to build identifier→concept mappings.

LSP/IDE data            Type information, call graphs, import
                        resolution. Use for richer encoding
                        when available.
```

### 6.3 Codebook Build Pipeline

```bash
# Step 1: Download and parse WordNet
python build_codebook.py --source wordnet --output codebook_wordnet.json

# Step 2: Expand with ConceptNet
python build_codebook.py --source conceptnet --base codebook_wordnet.json --output codebook_expanded.json

# Step 3: Fill gaps with Datamuse
python build_codebook.py --source datamuse --base codebook_expanded.json --output codebook_filled.json

# Step 4: Add sentiment data from SentiWordNet
python build_codebook.py --source sentiwordnet --base codebook_filled.json --output codebook_shaded.json

# Step 5: Add code tokens from tree-sitter grammars
python build_codebook.py --source treesitter --languages python,javascript,typescript,rust,go,java,c,cpp --base codebook_shaded.json --output codebook_full.json

# Step 6: Validate and version
python build_codebook.py --validate --stats --version 1.0.0 --output codebook_v1.0.0.json
```

### 6.4 Codebook Format

```json
{
  "version": "1.0.0",
  "created": "2026-04-22T00:00:00Z",
  "stats": {
    "total_entries": 487293,
    "domains": 25,
    "categories": 312,
    "concepts": 4096,
    "sources": ["wordnet-3.1", "conceptnet-5.9", "datamuse", "sentiwordnet-3.0", "treesitter"]
  },
  "domains": {
    "EM": { "name": "Emotion", "type": "text", "categories": 7, "concepts": 48 },
    "CF": { "name": "Control Flow", "type": "code", "categories": 7, "concepts": 36 }
  },
  "entries": {
    "happy": { "d": "EM", "c": 0, "n": 1, "s": { "p": 3, "f": 1, "i": 1 } },
    "joyful": { "d": "EM", "c": 0, "n": 1, "s": { "p": 3, "f": 1, "i": 2 } },
    "if": { "d": "CF", "c": 1, "n": 1, "s": { "p": 1, "f": 1, "i": 1 } },
    "switch": { "d": "CF", "c": 1, "n": 2, "s": { "p": 1, "f": 1, "i": 1 } },
    "async": { "d": "CF", "c": 5, "n": 1, "s": { "p": 1, "f": 2, "i": 1 } }
  }
}
```

The `"s"` field contains shade hints from the source data (SentiWordNet polarity, corpus formality scores, etc.) which override the heuristic shade computation.

---

## 7. Encoder Pipeline — Detailed Specification

### 7.1 Text Encoder

```
Input:  "The terrified soldiers retreated from the burning village"
                    │
        ┌───────────┴───────────┐
        │     TOKENIZE          │  Split on whitespace + punctuation
        │  → ["The","terrified",│  Remove punctuation tokens
        │    "soldiers", ...]   │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │     STOP WORD FILTER  │  Remove: the, from, a, an, is, etc.
        │  → ["terrified",     │  ~150 English stop words
        │    "soldiers", ...]   │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │     LEMMATIZE         │  "soldiers"→"soldier"
        │  → ["terrified",     │  "retreated"→"retreat"
        │    "soldier", ...]    │  "burning"→"burn"
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │     CODEBOOK LOOKUP   │  O(1) hash map lookup per token
        │  → [EM004, PE310,     │  Unknown tokens → unknowns list
        │     AC004, NA41C, ...]│
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │     SHADE COMPUTE     │  Per-token shade byte
        │  + CONTEXT MODIFY     │  Check neighbors for adverbs,
        │  → [EM004:C0,         │  negation, formality markers
        │     PE310:55, ...]    │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │     STRAND ASSEMBLE   │  Join codons into strand string
        │  → "EM004:C0·PE310:55│  Also output binary form for storage
        │    ·AC004:51·NA41C:55"│
        └───────────────────────┘

Output: {
  codons: [{word, codon, shade, domain}, ...],
  strand: "EM004:C0·PE310:55·AC004:51·NA41C:55",
  binary: <Buffer 45 4D 00 04 C0 50 45 03 10 55 ...>,
  unknowns: [],
  byte_size: 20
}
```

### 7.2 Code Encoder

Code encoding uses AST-aware tokenization. The encoder must handle two interleaved streams: **structural tokens** (keywords, operators, control flow) and **semantic tokens** (identifiers, string literals, comments).

```
Input:  async function fetchUser(id) {
          const response = await fetch(`/api/users/${id}`);
          if (!response.ok) throw new Error('Not found');
          return response.json();
        }

Step 1: PARSE AST (via tree-sitter)
  → FunctionDeclaration(async=true)
    ├── name: "fetchUser"
    ├── params: ["id"]
    └── body:
        ├── VariableDeclaration(const)
        │   └── AwaitExpression
        │       └── CallExpression("fetch", template_literal)
        ├── IfStatement
        │   ├── condition: UnaryExpression(!, member("response","ok"))
        │   └── consequent: ThrowStatement(NewExpression("Error", "Not found"))
        └── ReturnStatement
            └── CallExpression(member("response","json"))

Step 2: EXTRACT SEMANTIC TOKENS

  Structural tokens (→ code domains):
    async           → CF5.01  (Control Flow → Async → await/async)
    function        → MD0.01  (Module → declaration)
    const           → TS0.03  (Type System → declaration → const)
    await           → CF5.01  (Control Flow → Async → await)
    fetch           → IO1.01  (IO → Network → fetch)
    if              → CF1.01  (Control Flow → Conditional → if)
    throw           → ER0.03  (Error → Handling → throw)
    new Error       → ER0.01  (Error → Handling → throw) + DS0.07 (Data → composite)
    return          → CF4.04  (Control Flow → Branching → return)
    .json()         → FM0.01  (Format → JSON)

  Semantic tokens (→ text domains via identifier splitting):
    fetchUser       → split: "fetch" + "User" → AC7.2C (Action→search→find) + PE0.01 (Person)
    response        → CM0.02 (Communication → message)
    ok              → QU2.08 (Quality → good)
    id              → OB9.26 (Object → identifier/key)

Step 3: ASSEMBLE STRAND
  → "CF501:55·MD001:55·TS003:55·CF501:55·IO101:55·CF101:55·ER003:55·CF404:55·FM001:55·AC72C:55·PE001:55·CM002:55·QU208:57"

Step 4: OUTPUT
  {
    strand: "CF501:55·MD001:55·TS003:55·...",
    byte_size: 52,
    language: "javascript",
    structural_density: 0.69,  // ratio of code tokens to text tokens
    patterns_detected: ["async-await", "error-throw", "api-fetch"]
  }
```

### 7.3 Identifier Splitting

Code identifiers carry semantic meaning in their names. The encoder must split them into meaningful parts:

```
camelCase:     fetchUserData    → ["fetch", "user", "data"]
snake_case:    get_active_users → ["get", "active", "users"]
PascalCase:    HttpResponseCode → ["http", "response", "code"]
SCREAMING:     MAX_RETRY_COUNT  → ["max", "retry", "count"]
kebab-case:    user-profile-api → ["user", "profile", "api"]
abbreviated:   getDBConn        → ["get", "db", "conn"] → expand: ["get", "database", "connection"]
```

Common abbreviations should be expanded via a lookup table:

```json
{
  "db": "database", "conn": "connection", "req": "request", "res": "response",
  "msg": "message", "btn": "button", "cfg": "config", "env": "environment",
  "auth": "authentication", "admin": "administrator", "api": "interface",
  "url": "address", "err": "error", "fn": "function", "cb": "callback",
  "ctx": "context", "src": "source", "dst": "destination", "tmp": "temporary",
  "prev": "previous", "curr": "current", "idx": "index", "len": "length",
  "num": "number", "str": "string", "int": "integer", "bool": "boolean",
  "obj": "object", "arr": "array", "elem": "element", "attr": "attribute",
  "param": "parameter", "arg": "argument", "val": "value", "ref": "reference"
}
```

---

## 8. Comparator — Strand Alignment Algorithm

### 8.1 Pairwise Alignment

The comparator uses a greedy best-match alignment (similar to local sequence alignment in bioinformatics):

```python
def compare_strands(strand_a: Strand, strand_b: Strand) -> ComparisonResult:
    matches = []
    used_b = set()
    total_score = 0.0

    for codon_a in strand_a.codons:
        best_match = None
        best_score = 0.0

        for j, codon_b in enumerate(strand_b.codons):
            if j in used_b:
                continue

            score = 0.0
            if codon_a.domain == codon_b.domain:
                score += 0.25                          # Domain match
                if codon_a.category == codon_b.category:
                    score += 0.15                      # Category match
                    if codon_a.concept == codon_b.concept:
                        score += 0.35                  # Concept match
                        shade_diff = abs(codon_a.shade - codon_b.shade) / 255
                        score += (1 - shade_diff) * 0.25  # Shade proximity

            if score > best_score:
                best_score = score
                best_match = (j, codon_b, score)

        if best_match and best_match[2] > 0:
            used_b.add(best_match[0])
            matches.append(Match(a=codon_a, b=best_match[1], score=best_match[2]))
            total_score += best_match[2]

    max_len = max(len(strand_a.codons), len(strand_b.codons))
    overall = total_score / max_len if max_len > 0 else 0.0

    return ComparisonResult(
        score=overall,          # 0.0 to 1.0
        matches=matches,        # Per-codon alignment map
        explanation=build_explanation(matches, max_len)
    )
```

### 8.2 Scoring Thresholds

```
Score Range    Interpretation           Example
──────────     ──────────────           ──────────────────────────
0.90 - 1.00    Near-identical meaning   "happy dog" ↔ "joyful puppy"
0.70 - 0.89    Strong similarity        "happy dog" ↔ "cheerful hound"
0.40 - 0.69    Moderate relatedness     "happy dog" ↔ "excited cat"
0.15 - 0.39    Weak/domain overlap      "happy dog" ↔ "sad dog"
0.00 - 0.14    Unrelated                "happy dog" ↔ "SQL database"
```

### 8.3 Code-Specific Comparison Modifiers

When comparing code strands, apply these additional rules:

1. **Structural weight**: Code domain codons (CF, DS, TS, OP, IO, ER, PT, MD, TE, AP, IN) get 1.5x weight in scoring because structural similarity matters more than identifier similarity in code.
2. **Pattern matching**: If both strands contain the same detected pattern (e.g., "async-await", "error-handling", "CRUD"), add a 0.1 bonus to overall score.
3. **Order sensitivity**: For code, codon order matters more than for text. Apply a position-proximity bonus of up to 0.05 when matched codons appear in similar relative positions.

---

## 9. Index and Search

### 9.1 Storage

Strands are stored as simple byte arrays or strings. No specialized vector database required.

```sql
-- PostgreSQL example
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    strand      BYTEA NOT NULL,          -- Binary strand
    strand_text TEXT NOT NULL,            -- Human-readable strand
    domain_sig  TEXT NOT NULL,            -- Domain signature for pre-filtering
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Domain signature = sorted unique domains, e.g., "AC,EM,NA"
-- Use for fast pre-filtering before full alignment
CREATE INDEX idx_domain_sig ON documents(domain_sig);

-- For SQL LIKE searches on strand text
CREATE INDEX idx_strand_trgm ON documents USING gin(strand_text gin_trgm_ops);
```

### 9.2 Search Algorithm

```python
def search(index: list[IndexEntry], query: str, top_k: int = 10) -> list[SearchResult]:
    query_strand = encode(query)
    query_domains = set(c.domain for c in query_strand.codons)

    # Phase 1: Pre-filter by domain overlap (fast)
    candidates = [
        entry for entry in index
        if query_domains & entry.domains  # Set intersection
    ]

    # Phase 2: Full alignment on candidates
    results = []
    for entry in candidates:
        comparison = compare_strands(query_strand, entry.strand)
        results.append(SearchResult(entry=entry, score=comparison.score, explanation=comparison.explanation))

    # Phase 3: Sort and return top-k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
```

### 9.3 Performance Targets

```
Corpus Size     Search Latency (pre-filtered)    Memory
──────────      ────────────────────────         ──────
1K documents    < 1ms                            ~40KB strands
10K documents   < 10ms                           ~400KB strands
100K documents  < 100ms                          ~4MB strands
1M documents    < 500ms (with domain indexing)    ~40MB strands
10M documents   < 2s (with sharding)             ~400MB strands

Compare: 1M documents with embeddings = ~6GB of float vectors
         1M documents with strands    = ~40MB of byte sequences
```

---

## 10. MCP Server / Tool Interface

Semantic Strands should be deployable as an MCP server exposable to any LLM or agent framework:

### 10.1 Tools

```json
{
  "tools": [
    {
      "name": "strand_encode",
      "description": "Encode text or code into a semantic strand",
      "parameters": {
        "input": "string — text or code to encode",
        "mode": "string — 'text' | 'code' | 'auto'",
        "language": "string? — programming language (for code mode)"
      },
      "returns": {
        "strand": "string — codon:shade sequence",
        "codons": "array — detailed codon breakdown",
        "unknowns": "array — tokens not in codebook",
        "byte_size": "int — strand size in bytes"
      }
    },
    {
      "name": "strand_compare",
      "description": "Compare two strands for semantic similarity",
      "parameters": {
        "strand_a": "string — first strand or text to encode",
        "strand_b": "string — second strand or text to encode"
      },
      "returns": {
        "score": "float — 0.0 to 1.0",
        "matches": "array — per-codon alignment map",
        "explanation": "string — human-readable match explanation"
      }
    },
    {
      "name": "strand_index",
      "description": "Add a document/code to the search index",
      "parameters": {
        "id": "string — document identifier",
        "content": "string — text or code to index",
        "metadata": "object? — optional metadata to store"
      }
    },
    {
      "name": "strand_search",
      "description": "Search the index for semantically similar content",
      "parameters": {
        "query": "string — search query (text or code)",
        "top_k": "int? — number of results (default 10)",
        "threshold": "float? — minimum score (default 0.0)",
        "domains": "array? — filter to specific domains"
      },
      "returns": {
        "results": "array — ranked results with scores and explanations"
      }
    },
    {
      "name": "strand_inspect",
      "description": "Inspect the codebook entry for a word/token",
      "parameters": {
        "token": "string — word or code token to look up"
      },
      "returns": {
        "found": "bool",
        "codon": "string",
        "domain": "string",
        "category": "string",
        "concept": "string",
        "shade": "object — intensity, abstraction, formality, polarity",
        "synonyms": "array — other words with same codon"
      }
    }
  ]
}
```

### 10.2 Deployment Options

```
OPTION              COMMAND                              USE CASE
──────────          ──────────────────────────           ────────────
CLI tool            strand encode "hello world"          Scripts, pipelines
Python package      from strands import encode           Application integration
MCP server          strand-server --port 3847            LLM tool use
REST API            POST /api/v1/encode                  Web services
WASM module         import { encode } from 'strands'     Browser / edge
SQLite extension    SELECT strand_encode(text)           Database-native
```

---

## 11. Implementation Priorities

### Phase 1: Core Engine (MVP)

- [ ] Codebook builder: WordNet → codebook JSON pipeline
- [ ] Text encoder with lemmatization and stop word filtering
- [ ] Shade byte computation with sentiment/formality/intensity lexicons
- [ ] Pairwise strand comparator with alignment scoring
- [ ] In-memory index with brute-force search
- [ ] CLI tool: `strand encode`, `strand compare`, `strand search`
- [ ] Test suite: synonym/antonym/unrelated benchmarks against known word-pair datasets (SimLex-999, WordSim-353)

### Phase 2: Code Support

- [ ] Tree-sitter integration for AST-aware tokenization
- [ ] Code domain codebook from language grammars
- [ ] Identifier splitting and abbreviation expansion
- [ ] Multi-language support: Python, JavaScript, TypeScript, Rust, Go, Java, C/C++
- [ ] Code search benchmarks against CodeSearchNet dataset

### Phase 3: Scale and Distribution

- [ ] MCP server implementation
- [ ] PostgreSQL/SQLite storage adapters with domain-signature indexing
- [ ] Codebook versioning and update mechanism
- [ ] Datamuse/ConceptNet gap-filling for unknown tokens at runtime
- [ ] REST API server
- [ ] WASM build for browser deployment

### Phase 4: Advanced Features

- [ ] Context-aware shade modification (negation, intensifiers, formality markers)
- [ ] Document-level strand summarization (compress long strands)
- [ ] Cross-language code comparison (Python function ↔ equivalent Rust function)
- [ ] Strand clustering for topic modeling
- [ ] Embedding migration tool: convert existing vector indices to strand indices

---

## 12. Testing and Validation

### 12.1 Word-Level Benchmarks

Test against established word similarity datasets:

```
DATASET          PAIRS   METRIC              TARGET
────────         ─────   ──────              ──────
SimLex-999       999     Spearman ρ ≥ 0.35   (embeddings typically achieve 0.40-0.50)
WordSim-353      353     Spearman ρ ≥ 0.55   (embeddings typically achieve 0.60-0.75)
MEN-3000         3000    Spearman ρ ≥ 0.50   (embeddings typically achieve 0.70-0.80)
RG-65            65      Spearman ρ ≥ 0.70   (embeddings typically achieve 0.75-0.85)
```

Note: Strands will likely underperform embeddings on these benchmarks because they sacrifice continuous similarity for interpretability and efficiency. The target is "good enough" correlation while delivering 1000x+ size reduction and full interpretability.

### 12.2 Sentence-Level Benchmarks

```
DATASET              METRIC              TARGET
────────             ──────              ──────
STS Benchmark        Spearman ρ ≥ 0.40   Sentence similarity
SICK                 Spearman ρ ≥ 0.45   Semantic textual similarity
```

### 12.3 Code Benchmarks

```
DATASET              METRIC              TARGET
────────             ──────              ──────
CodeSearchNet        MRR ≥ 0.25          Code search (natural language → code)
BigCloneBench        F1 ≥ 0.50           Code clone detection
```

### 12.4 Automated Test Categories

Every build must pass:

```
Category         Count    Assertion
────────         ─────    ─────────
Synonym pairs     200+    score ≥ 0.70
Near-synonym      100+    0.40 ≤ score ≤ 0.90
Antonym pairs     100+    0.15 ≤ score ≤ 0.45
Unrelated pairs   100+    score ≤ 0.15
Code synonyms      50+    score ≥ 0.60  (e.g., for-loop ↔ forEach)
Cross-domain        50+   score ≤ 0.10  (e.g., "happy" ↔ "async")
```

---

## 13. File Formats

### 13.1 Strand Binary Format

```
Header (8 bytes):
  [0-1]  Magic: 0x5353 ("SS")
  [2]    Version: 0x01
  [3]    Flags: bit 0 = has_metadata, bit 1 = code_mode
  [4-5]  Codon count (uint16, big-endian)
  [6-7]  Reserved

Body (4 bytes per codon):
  [0]    Domain (uint8, index into domain table)
  [1]    Category (uint8)
  [2]    Concept (uint8)
  [3]    Shade (uint8)

Optional metadata (if flag bit 0 set):
  [0-1]  Metadata length (uint16)
  [2+]   JSON metadata bytes
```

### 13.2 Strand Text Format

Human-readable, used for debugging, logging, and text-based storage:

```
EM001:5D·AC402:71·NA312:44·PE30E:55
```

Rules:
- Domain: 2 uppercase letters
- Category: 1 hex digit
- Concept: 2 hex digits
- Separator: colon
- Shade: 2 hex digits
- Codon separator: middle dot (·) or pipe (|) or comma
- Whitespace is ignored

---

## 14. Codebook Governance

### 14.1 Versioning

Codebooks follow semantic versioning:
- **Major**: Domain restructuring, codon remapping (requires re-encoding)
- **Minor**: New concepts/words added (backward compatible)
- **Patch**: Typo fixes, shade adjustments (backward compatible)

### 14.2 Human Editability

The codebook is a JSON file that humans can inspect and edit. This is a core design principle — unlike embedding models, the codebook is transparent. If "algorithm" is miscategorized, a human can fix it with a text editor.

### 14.3 Extension Mechanism

Users can create domain-specific codebook extensions without modifying the base codebook:

```json
{
  "extends": "codebook_v1.0.0.json",
  "name": "medical-extension",
  "entries": {
    "tachycardia": { "d": "BD", "c": 1, "n": 99, "s": { "p": 0, "f": 3, "i": 2 } },
    "bradycardia": { "d": "BD", "c": 1, "n": 98, "s": { "p": 0, "f": 3, "i": 1 } }
  }
}
```

---

## 15. Known Limitations and Tradeoffs

1. **Lower fidelity than embeddings** — Strands sacrifice continuous similarity for discretized categorical encoding. "happy" and "joyful" share an exact codon, but fine-grained differences between "content" and "pleased" may be lost. The shade byte recovers some nuance but 8 bits cannot match 1536 floats.

2. **Codebook coverage** — Words not in the codebook produce unknowns. Mitigated by gap-filling from Datamuse/ConceptNet at runtime, but rare/novel words may fall through. Embeddings handle unknown words via subword tokenization.

3. **Word sense disambiguation** — "bank" (financial) vs "bank" (river) map to a single codon. The codebook assigns the most common sense. Future work: use surrounding codons to disambiguate.

4. **No transfer learning** — Embeddings capture statistical relationships from massive corpora that codebooks cannot replicate. Strands are taxonomy-based, not distribution-based.

5. **Maintenance burden** — The codebook must be maintained and updated. Embedding models are maintained by their providers. However, codebook updates are transparent and controllable.

### When to Use Strands vs Embeddings

| Use Strands When | Use Embeddings When |
|---|---|
| Storage/bandwidth is constrained | Maximum fidelity is critical |
| You need interpretable matching | You need multilingual zero-shot |
| You want vendor independence | You need subword handling |
| You run on constrained hardware | You have GPU infrastructure |
| You need human-auditable results | Fine-grained similarity matters |
| Your corpus is < 10M documents | Your corpus is > 100M documents |
| You want deterministic encoding | You need continuous similarity space |

---

## Appendix A: Quick Reference

```
Strand = sequence of (codon + shade) pairs
Codon  = 3-byte semantic address: Domain(1) + Category(1) + Concept(1)
Shade  = 1-byte nuance: Intensity(2b) + Abstraction(2b) + Formality(2b) + Polarity(2b)
Token  = codon:shade = 4 bytes total

Encoding:   text → tokenize → lemmatize → codebook lookup → shade → strand
Comparison: strand × strand → greedy alignment → score [0.0, 1.0]
Search:     query strand → domain pre-filter → align candidates → rank → top-k
Storage:    any database, any file format, even grep
```
