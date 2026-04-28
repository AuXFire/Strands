"""Assemble the final codebook JSON from cached build layers.

The build pipeline is split into deterministic, content-addressed layers
(see strands/build/cache.py). A change in one layer's input invalidates
only that layer and its downstream layers — small seed edits no longer
trigger full rebuilds.

  Layer 1: WordNet expansion         deterministic from seeds + WN
  Layer 2: morphology variants       deterministic from layer 1
  Layer 3: sentiment polarity hints  deterministic from layer 1 + SentiWN
  Layer 4: codon adjacency graph     deterministic from layer 1 + Numberbatch
            (built by scripts/build_codon_adjacency.py)

The merged JSON itself is cheap to produce from these layers (<1 s).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from strands.build.cache import (
    SENTIWORDNET_VERSION,
    WORDNET_VERSION,
    get_or_build,
    hash_seeds,
)
from strands.build.frequency_filter import (
    formality_from_frequency,
    is_common,
)
from strands.build.morphology import variants_for, wordnet_irregular_forms
from strands.build.seeds import ALL_CODE_SEEDS, ALL_SEEDS
from strands.build.sentiwordnet import polarity_bits
from strands.build.wordnet_builder import expand_seeds_with_pos
from strands.codon import DOMAIN_NAMES

CODEBOOK_VERSION = "0.1.0"


def _intensity_for(word: str) -> int:
    return 1


# --- Layer builders --------------------------------------------------------


def _build_wordnet_layer() -> dict:
    """Layer 1 — full WordNet expansion. The slow one (~20 s)."""
    primary, pos_map, alts, synsets = expand_seeds_with_pos(ALL_SEEDS)
    return {
        "primary": {w: list(c) for w, c in primary.items()},
        "alts": {w: [list(c) for c in cs] for w, cs in alts.items()},
        "synsets": dict(synsets),
        "pos": dict(pos_map),
    }


def _build_morphology_layer(wn_payload: dict) -> dict:
    """Layer 2 — rule-based inflection list ``[(variant, base, pos), ...]``."""
    pos_map = wn_payload["pos"]
    edges: list[list[str]] = []
    base_words = sorted(pos_map.keys(), key=lambda w: (len(w), w))
    for word in base_words:
        for pos in pos_map.get(word, []):
            for variant in variants_for(word, pos):
                edges.append([variant, word, pos])
    irregular = wordnet_irregular_forms()
    return {
        "edges": edges,
        "irregular": {k: list(v) for k, v in irregular.items()},
    }


def _build_sentiment_layer(wn_payload: dict) -> dict:
    """Layer 3 — polarity bits + formality bits per word.

    Both channels are pure deterministic functions of the word + its
    SentiWordNet/wordfreq lookups, so they belong in this cache. Caching
    here lets the merge step be a simple dict compose with no per-word
    Python-side computation, which is what makes warm rebuilds fast.
    """
    polarity: dict[str, int] = {}
    formality: dict[str, int] = {}
    for word in wn_payload["primary"].keys():
        polarity[word] = polarity_bits(word)
        formality[word] = formality_from_frequency(word)
    return {"polarity": polarity, "formality": formality}


# --- Merge -----------------------------------------------------------------


def _merge_layers(
    wn: dict,
    morph: dict,
    sent: dict,
    *,
    frequency_threshold: float | None,
    include_inflections: bool,
) -> dict[str, dict]:
    """Compose the final word -> entry dict from cached layers."""
    seed_words: set[str] = {w.lower() for words in ALL_SEEDS.values() for w in words}
    primary: dict[str, list] = wn["primary"]
    synsets: dict[str, str] = wn["synsets"]
    polarity: dict[str, int] = sent["polarity"]
    formality: dict[str, int] = sent.get("formality", {})

    entries: dict[str, dict] = {}
    for word, (domain_code, category, concept) in sorted(primary.items()):
        if (
            frequency_threshold is not None
            and word not in seed_words
            and not is_common(word, frequency_threshold)
        ):
            continue

        entry: dict = {
            "d": domain_code,
            "c": category,
            "n": concept,
            "s": {
                "p": polarity.get(word, 1),
                "f": formality.get(word) if word in formality
                     else formality_from_frequency(word),
                "i": _intensity_for(word),
            },
        }
        if (syn := synsets.get(word)):
            entry["syn"] = syn
        entries[word] = entry

    if include_inflections:
        # Apply rule-based inflectional variants from cached morphology layer.
        # Variants inherit codon, shade, and synset from their base lemma.
        # The ``rel`` field is not populated here — the build_strand_relations
        # step (which runs after this) propagates it to variants.
        for variant, base, _pos in morph["edges"]:
            if variant in seed_words:
                continue
            base_entry = entries.get(base)
            if base_entry is None:
                continue
            new_entry = {
                "d": base_entry["d"],
                "c": base_entry["c"],
                "n": base_entry["n"],
                "s": dict(base_entry["s"]),
            }
            if "syn" in base_entry:
                new_entry["syn"] = base_entry["syn"]
            entries[variant] = new_entry

        # WordNet's irregular-form exception lists (e.g. went -> go).
        # Iterate in alphabetical order — chains like senti -> sent -> send
        # depend on processing order. Sorting makes the result deterministic
        # regardless of whether morph["irregular"] came from a fresh build
        # or a JSON-cached round trip.
        for inflected in sorted(morph["irregular"]):
            if inflected in seed_words:
                continue
            for lemma in morph["irregular"][inflected]:
                base = entries.get(lemma)
                if base is not None:
                    entries[inflected] = {
                        "d": base["d"],
                        "c": base["c"],
                        "n": base["n"],
                        "s": dict(base["s"]),
                    }
                    break

    return entries


def build_code_entries() -> dict[str, dict]:
    """Code-domain entries — direct from seeds, no WordNet expansion."""
    out: dict[str, dict] = {}
    for (domain_code, category, concept), words in ALL_CODE_SEEDS.items():
        for word in words:
            w = word.lower().strip()
            if not w or w in out:
                continue
            out[w] = {
                "d": domain_code,
                "c": category,
                "n": concept,
                "s": {"p": 1, "f": 2, "i": 1},
            }
    return out


def build(
    *,
    frequency_threshold: float | None = None,
    include_inflections: bool = True,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    invalidate: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Build the codebook dict via the layered cache.

    ``invalidate`` is a list of layer names ("wn", "morph", "sent") whose
    cached artifacts will be deleted before this build, forcing recompute.
    Pass ``["all"]`` to drop everything.
    """
    seeds_hash = hash_seeds(ALL_SEEDS, ALL_CODE_SEEDS)
    wn_input_hash = f"{seeds_hash}-{WORDNET_VERSION}"

    invalidate = set(invalidate or [])
    if "all" in invalidate:
        invalidate = {"wn", "morph", "sent"}

    if use_cache:
        if "wn" in invalidate:
            from strands.build.cache import LayerCache
            LayerCache(cache_dir, "wn").clear()
        if "morph" in invalidate:
            from strands.build.cache import LayerCache
            LayerCache(cache_dir, "morph").clear()
        if "sent" in invalidate:
            from strands.build.cache import LayerCache
            LayerCache(cache_dir, "sent").clear()

        wn = get_or_build(
            "wn", wn_input_hash, _build_wordnet_layer,
            cache_dir=cache_dir,
            sources={"seeds_hash": seeds_hash, "wordnet": WORDNET_VERSION},
            verbose=verbose,
        )
        wn_hash = wn_input_hash  # downstream layers key off the WN input hash
        morph = get_or_build(
            "morph", wn_hash, lambda: _build_morphology_layer(wn),
            cache_dir=cache_dir,
            sources={"wn_hash": wn_hash},
            verbose=verbose,
        )
        sent_input_hash = f"{wn_hash}-{SENTIWORDNET_VERSION}"
        sent = get_or_build(
            "sent", sent_input_hash, lambda: _build_sentiment_layer(wn),
            cache_dir=cache_dir,
            sources={"wn_hash": wn_hash, "sentiwordnet": SENTIWORDNET_VERSION},
            verbose=verbose,
        )
    else:
        wn = _build_wordnet_layer()
        morph = _build_morphology_layer(wn)
        sent = _build_sentiment_layer(wn)

    entries = _merge_layers(
        wn, morph, sent,
        frequency_threshold=frequency_threshold,
        include_inflections=include_inflections,
    )

    domains_meta = {}
    for code, name in DOMAIN_NAMES.items():
        cats = {(c, n) for (d, c, n) in ALL_SEEDS.keys() if d == code}
        if cats:
            entry_count = sum(1 for v in entries.values() if v["d"] == code)
            domains_meta[code] = {
                "name": name,
                "type": "text",
                "categories": len({c for c, _ in cats}),
                "concepts": len(cats),
                "entries": entry_count,
            }

    code_entries = build_code_entries()

    return {
        "version": CODEBOOK_VERSION,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "total_entries": len(entries),
            "code_entries": len(code_entries),
            "domains": len(domains_meta),
            "categories": sum(d["categories"] for d in domains_meta.values()),
            "concepts": sum(d["concepts"] for d in domains_meta.values()),
            "sources": ["wordnet-3.1", "sentiwordnet-3.0", "wordfreq"],
            "seeds_hash": seeds_hash,
        },
        "domains": domains_meta,
        "entries": entries,
        "code_entries": code_entries,
    }


def write(
    output_path: Path | str,
    *,
    frequency_threshold: float | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    invalidate: list[str] | None = None,
    preserve_relations: bool = True,
) -> dict:
    """Write the codebook JSON.

    When ``preserve_relations`` is True (default), any ``rel`` fields in
    an existing output file are carried over — so a seed-only rebuild
    doesn't drop the ConceptNet-derived per-word relations (which take
    ~1 minute to recompute via build_strand_relations.py).
    """
    codebook = build(
        frequency_threshold=frequency_threshold,
        use_cache=use_cache,
        cache_dir=cache_dir,
        invalidate=invalidate,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if preserve_relations and output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_entries = existing.get("entries", {})
            preserved = 0
            for word, raw in codebook.get("entries", {}).items():
                old = existing_entries.get(word)
                if old and "rel" in old:
                    raw["rel"] = old["rel"]
                    preserved += 1
            if preserved:
                codebook.setdefault("stats", {})["preserved_rel_entries"] = preserved
        except (OSError, json.JSONDecodeError):
            pass

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    return codebook
