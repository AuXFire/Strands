"""Assemble the final codebook JSON.

Pipeline:
  seeds → wordnet expansion (full vocab) → sentiwordnet polarity → JSON

Frequency information is recorded as a formality hint per entry but is no
longer used to drop entries — every classified WordNet lemma is included.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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


def build(
    *,
    frequency_threshold: float | None = None,
    include_inflections: bool = True,
) -> dict:
    """Build the codebook dict.

    `frequency_threshold` is optional. When None, every classified WordNet
    lemma is included (full vocabulary). When set, non-seed entries with
    Zipf frequency below the threshold are dropped — useful for producing
    smaller diagnostic codebooks.

    `include_inflections` adds rule-based inflectional variants (plurals,
    -ed/-ing, comparatives) plus WordNet's irregular-form exception lists.
    """
    seed_map, word_to_pos = expand_seeds_with_pos(ALL_SEEDS)
    seed_words: set[str] = {w.lower() for words in ALL_SEEDS.values() for w in words}

    entries: dict[str, dict] = {}
    for word, (domain_code, category, concept) in sorted(seed_map.items()):
        if (
            frequency_threshold is not None
            and word not in seed_words
            and not is_common(word, frequency_threshold)
        ):
            continue

        polarity = polarity_bits(word)
        formality = formality_from_frequency(word)
        intensity = _intensity_for(word)

        entries[word] = {
            "d": domain_code,
            "c": category,
            "n": concept,
            "s": {
                "p": polarity,
                "f": formality,
                "i": intensity,
            },
        }

    if include_inflections:
        # Inflectional variants overwrite any conflicting WordNet entries so
        # that, e.g., "running" inherits from the verb "run" rather than
        # WordNet's noun sense ("the running of a business").
        # Seeds are protected — they always win.
        # Snapshot the WordNet+seed bases up front so we don't iterate over
        # the inflections we just generated (which would create "happiers").
        wordnet_bases = list(entries.keys())
        for word in sorted(wordnet_bases, key=lambda w: (len(w), w)):
            base_entry = entries[word]
            for pos in word_to_pos.get(word, []):
                for variant in variants_for(word, pos):
                    if variant in seed_words:
                        continue
                    entries[variant] = {
                        "d": base_entry["d"],
                        "c": base_entry["c"],
                        "n": base_entry["n"],
                        "s": dict(base_entry["s"]),
                    }

        # WordNet's irregular-form exception lists (e.g. went -> go).
        for inflected, lemmas in wordnet_irregular_forms().items():
            if inflected in seed_words:
                continue
            for lemma in lemmas:
                base = entries.get(lemma)
                if base is not None:
                    entries[inflected] = {
                        "d": base["d"],
                        "c": base["c"],
                        "n": base["n"],
                        "s": dict(base["s"]),
                    }
                    break

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
        },
        "domains": domains_meta,
        "entries": entries,
        "code_entries": code_entries,
    }


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
                "s": {"p": 1, "f": 2, "i": 1},  # neutral, slightly formal
            }
    return out


def write(output_path: Path | str, *, frequency_threshold: float | None = None) -> dict:
    codebook = build(frequency_threshold=frequency_threshold)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    return codebook
