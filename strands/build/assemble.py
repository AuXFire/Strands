"""Assemble the final codebook JSON.

Pipeline:
  seeds → wordnet expansion → frequency filter → sentiwordnet polarity → JSON
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from strands.build.frequency_filter import (
    formality_from_frequency,
    is_common,
)
from strands.build.seeds import ALL_SEEDS
from strands.build.sentiwordnet import polarity_bits
from strands.build.wordnet_builder import expand_seeds
from strands.codon import DOMAIN_NAMES

CODEBOOK_VERSION = "0.1.0"


def _intensity_for(word: str) -> int:
    # Heuristic: longer words tend to be less intense in everyday speech.
    # Default to 1 (normal). Specific overrides can be added later.
    return 1


def _abstraction_for(word: str) -> int:
    return min(3, len(word) // 4)


def build(*, frequency_threshold: float = 2.0) -> dict:
    seed_map = expand_seeds(ALL_SEEDS, hyponym_depth=1)

    entries: dict[str, dict] = {}
    for word, (domain_code, category, concept) in sorted(seed_map.items()):
        # Always include direct seed words (even if rare); filter only
        # the WordNet-expanded long tail by frequency.
        is_seed = word in {w for words in ALL_SEEDS.values() for w in words}
        if not is_seed and not is_common(word, frequency_threshold):
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

    domains_meta = {}
    for code, name in DOMAIN_NAMES.items():
        cats = {(c, n) for (d, c, n) in ALL_SEEDS.keys() if d == code}
        if cats:
            domains_meta[code] = {
                "name": name,
                "type": "text",
                "categories": len({c for c, _ in cats}),
                "concepts": len(cats),
            }

    return {
        "version": CODEBOOK_VERSION,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "total_entries": len(entries),
            "domains": len(domains_meta),
            "categories": sum(d["categories"] for d in domains_meta.values()),
            "concepts": sum(d["concepts"] for d in domains_meta.values()),
            "sources": ["wordnet-3.1", "sentiwordnet-3.0", "wordfreq"],
        },
        "domains": domains_meta,
        "entries": entries,
    }


def write(output_path: Path | str, *, frequency_threshold: float = 2.0) -> dict:
    codebook = build(frequency_threshold=frequency_threshold)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, sort_keys=True)
    return codebook
