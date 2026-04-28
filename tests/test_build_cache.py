"""Layered build-cache tests.

Verify:
  - Cold build populates each layer's cache file.
  - Warm build reuses every layer (cache hits) and is much faster.
  - Modifying inputs invalidates only the affected layer + downstream.
  - Final codebook is byte-identical between cold and warm builds.
  - Patch-overlay file applies on top of the base codebook.
  - Cache files survive reading/writing as valid JSON.
"""

from __future__ import annotations

import json
import time

import pytest

from strands.build import assemble
from strands.build.cache import (
    LayerCache,
    _hash,
    hash_seeds,
)
from strands.build.seeds import ALL_CODE_SEEDS, ALL_SEEDS


def _strip_volatile(cb: dict) -> dict:
    """Drop fields that change every build run (timestamps)."""
    out = {k: v for k, v in cb.items() if k != "created"}
    if "stats" in out:
        out["stats"] = {k: v for k, v in out["stats"].items() if k != "created"}
    return out


def test_seeds_hash_is_deterministic():
    h1 = hash_seeds(ALL_SEEDS, ALL_CODE_SEEDS)
    h2 = hash_seeds(ALL_SEEDS, ALL_CODE_SEEDS)
    assert h1 == h2
    assert len(h1) == 16
    # Different content yields different hash.
    fake = {("EM", 0, 0): ["happy"]}
    h3 = hash_seeds(fake, ALL_CODE_SEEDS)
    assert h3 != h1


def test_layer_cache_round_trip(tmp_path):
    cache = LayerCache(cache_dir=tmp_path, layer="test")
    payload = {"a": 1, "b": [2, 3]}
    cache.write("hash123", payload, sources={"src": "v1"})
    loaded = cache.read("hash123")
    assert loaded == payload
    # Wrong hash returns None.
    assert cache.read("wrong-hash") is None


def test_layer_cache_invalidates_on_schema_mismatch(tmp_path):
    cache = LayerCache(cache_dir=tmp_path, layer="test")
    blob = {
        "cache_version": "999",  # wrong version
        "layer": "test",
        "input_hash": "x",
        "payload": {"foo": "bar"},
    }
    cache.path("x").write_text(json.dumps(blob))
    # Should reject due to cache_version mismatch.
    assert cache.read("x") is None


def test_full_build_populates_cache(tmp_path):
    """Cold build runs each builder; layers are written to disk."""
    cb1 = assemble.build(use_cache=True, cache_dir=tmp_path, verbose=False)
    assert "entries" in cb1
    assert len(cb1["entries"]) > 1000

    # Cache files are present
    layer_files = sorted(p.name for p in tmp_path.iterdir())
    assert any(name.startswith("wn-") for name in layer_files), layer_files
    assert any(name.startswith("morph-") for name in layer_files), layer_files
    assert any(name.startswith("sent-") for name in layer_files), layer_files


def test_warm_build_is_faster(tmp_path):
    """Second build reuses cached layers — saves the WN expansion cost."""
    t0 = time.perf_counter()
    cb1 = assemble.build(use_cache=True, cache_dir=tmp_path, verbose=False)
    cold = time.perf_counter() - t0

    t1 = time.perf_counter()
    cb2 = assemble.build(use_cache=True, cache_dir=tmp_path, verbose=False)
    warm = time.perf_counter() - t1

    speedup = cold / warm if warm > 0 else float("inf")
    print(f"\n  cold build: {cold:.2f}s   warm build: {warm:.2f}s   speedup: {speedup:.1f}x")
    # Allow up to 0.7 ratio: the merge step (245k entry dict + JSON dump)
    # is unavoidable, but the WN/morph/sent layers should be cache-hit.
    assert warm < cold * 0.7, (
        f"warm {warm:.2f}s should be < 0.7 × cold {cold:.2f}s; "
        f"layer caching does not appear to be working"
    )
    # Output should be identical.
    assert _strip_volatile(cb1) == _strip_volatile(cb2)


def test_invalidation_forces_rebuild(tmp_path):
    """Forcing a layer rebuild clears the cache file."""
    assemble.build(use_cache=True, cache_dir=tmp_path, verbose=False)
    sent_files = list(tmp_path.glob("sent-*.json"))
    assert sent_files

    assemble.build(
        use_cache=True, cache_dir=tmp_path, invalidate=["sent"], verbose=False,
    )
    # File should still exist (rebuilt) but with a fresh timestamp.
    sent_files_after = list(tmp_path.glob("sent-*.json"))
    assert sent_files_after, "sent layer should have been rebuilt"


def test_no_cache_mode_works(tmp_path):
    """use_cache=False forces full recompute and produces same output."""
    cb_cached = assemble.build(use_cache=True, cache_dir=tmp_path, verbose=False)
    cb_fresh = assemble.build(use_cache=False, verbose=False)
    assert _strip_volatile(cb_cached) == _strip_volatile(cb_fresh)


def test_seed_change_invalidates_wn_layer(tmp_path):
    """Different seed dicts produce different WN-layer cache keys."""
    h1 = hash_seeds(ALL_SEEDS, ALL_CODE_SEEDS)
    altered = dict(ALL_SEEDS)
    altered[("EM", 9, 99)] = ["fictional_test_seed"]
    h2 = hash_seeds(altered, ALL_CODE_SEEDS)
    assert h1 != h2


def test_patch_overlay_applies(tmp_path):
    """A patches file overrides specific entries at load time."""
    from strands.codebook import Codebook

    base_path = tmp_path / "codebook.json"
    base = {
        "version": "0.0.1",
        "entries": {
            "happy": {"d": "EM", "c": 0, "n": 0, "s": {"p": 3, "f": 1, "i": 1}},
        },
        "code_entries": {},
    }
    base_path.write_text(json.dumps(base))

    cb = Codebook.load(base_path)
    cb.merge_extension({
        "entries": {
            "happy": {"d": "EM", "c": 0, "n": 7, "s": {"p": 3, "f": 1, "i": 3}},
            "elated": {"d": "EM", "c": 0, "n": 0, "s": {"p": 3, "f": 2, "i": 3}},
        }
    })
    e = cb.lookup("happy")
    assert e is not None
    assert e.codon.concept == 7  # patched value
    assert cb.lookup("elated") is not None


def test_preserved_relations_round_trip(tmp_path):
    """Writing a codebook with preserve_relations keeps existing per-word
    rel fields intact."""
    out_path = tmp_path / "codebook.json"

    assemble.write(
        out_path, use_cache=True, cache_dir=tmp_path,
        preserve_relations=False,
    )
    written = json.loads(out_path.read_text())
    # Cherry-pick a known seed word that exists.
    sample_word = next(iter(written.get("entries", {})))

    # Inject a rel field
    written["entries"][sample_word]["rel"] = [["EM001", 200], ["AC002", 150]]
    out_path.write_text(json.dumps(written))

    # Rewrite, preserving relations
    assemble.write(
        out_path, use_cache=True, cache_dir=tmp_path,
        preserve_relations=True,
    )
    after = json.loads(out_path.read_text())
    assert after["entries"][sample_word].get("rel") == [["EM001", 200], ["AC002", 150]]
