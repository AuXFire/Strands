"""Build-cache layer: deterministic, content-addressed intermediate artifacts.

The codebook is built in five layers, each with its own cache file keyed
by the hash of its inputs. A change to one layer's input invalidates only
that layer and its downstream layers, not the entire pipeline.

  Layer 0  seeds            hash of seed dicts
  Layer 1  wordnet           inputs: seeds + WN version
  Layer 2  morphology        inputs: WN layer
  Layer 3  sentiment         inputs: WN layer + SentiWordNet version
  Layer 4  adjacency         inputs: WN layer + Numberbatch identifier

Each cache file is a small JSON with header (``input_hash``, source
versions, build timestamp) plus the layer's payload. A miss triggers
recomputation; a hit deserialises directly. Every layer is also a stable
artifact — hand-edits to a cache file produce a new content hash and
invalidate downstream layers, surfacing the change explicitly.

The merged codebook is *not* cached — it's cheap (<1 s) to recompose
from the layer files.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

CACHE_VERSION = "1"  # bump when the cache schema changes
DEFAULT_CACHE_DIR = Path(__file__).parent / "_cache"

# External dataset versions. Bump these when the upstream release changes.
WORDNET_VERSION = "wn-3.0"
SENTIWORDNET_VERSION = "swn-3.0"
NUMBERBATCH_VERSION = "nb-17.06.300"


def _hash(payload: Any) -> str:
    """Stable short hash of any JSON-serializable payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def hash_seeds(text_seeds: dict, code_seeds: dict) -> str:
    return _hash({"text": _normalize_seeds(text_seeds), "code": _normalize_seeds(code_seeds)})


def _normalize_seeds(seeds: dict) -> list:
    """Convert seed dict (tuple keys) to canonical sortable form."""
    return sorted(
        ([list(k) if isinstance(k, tuple) else k, sorted(v)] for k, v in seeds.items()),
        key=lambda x: str(x[0]),
    )


class LayerCache:
    """Content-addressed JSON cache for a single build layer."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        layer: str = "",
    ) -> None:
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.layer = layer

    def path(self, input_hash: str) -> Path:
        return self.cache_dir / f"{self.layer}-{input_hash}.json"

    def read(self, input_hash: str) -> dict | None:
        p = self.path(input_hash)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if blob.get("cache_version") != CACHE_VERSION:
            return None
        if blob.get("input_hash") != input_hash:
            return None
        return blob.get("payload")

    def write(self, input_hash: str, payload: Any, *, sources: dict | None = None) -> None:
        blob = {
            "cache_version": CACHE_VERSION,
            "layer": self.layer,
            "input_hash": input_hash,
            "sources": sources or {},
            "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": payload,
        }
        tmp = self.path(input_hash).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(blob, f, sort_keys=True)
        os.replace(tmp, self.path(input_hash))

    def clear(self) -> None:
        for p in self.cache_dir.glob(f"{self.layer}-*.json"):
            p.unlink()


def get_or_build(
    layer: str,
    input_hash: str,
    builder,
    *,
    cache_dir: Path | None = None,
    sources: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Return cached payload for (layer, input_hash) or compute via builder."""
    cache = LayerCache(cache_dir=cache_dir, layer=layer)
    cached = cache.read(input_hash)
    if cached is not None:
        if verbose:
            print(f"  [cache HIT ] {layer}-{input_hash}")
        return cached
    if verbose:
        print(f"  [cache MISS] {layer}-{input_hash}  building …")
    t0 = time.perf_counter()
    payload = builder()
    elapsed = time.perf_counter() - t0
    cache.write(input_hash, payload, sources=sources)
    if verbose:
        print(f"  [cache WRITE] {layer}-{input_hash}  ({elapsed:.1f}s)")
    return payload
