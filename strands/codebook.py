"""Codebook loader — O(1) word → codon lookup with ConceptNet-derived
related codons.

The codebook is a build-time artifact. The encoder uses it to stamp each
token with its primary codon, shade hint, and up to two related codons
(from ConceptNet). After encoding, the resulting strand is fully
self-contained — comparing two strands needs nothing but their bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from strands.codon import DOMAIN_CODES, NULL_CODON, Codon


@dataclass(frozen=True, slots=True)
class CodebookEntry:
    word: str
    codon: Codon
    domain_code: str
    shade_hint: dict
    related: tuple[tuple[Codon, int], ...] = ()
    synset: str = ""


class Codebook:
    def __init__(self, data: dict, *, base_dir: Path | None = None):
        self.version: str = data.get("version", "unknown")
        self.stats: dict = data.get("stats", {})
        self.domains: dict = data.get("domains", {})
        self._entries: dict[str, dict] = data.get("entries", {})
        self._code_entries: dict[str, dict] = data.get("code_entries", {})

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def code_size(self) -> int:
        return len(self._code_entries)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._entries

    def _decode_codon(self, codon_repr) -> Codon | None:
        """Decode either a string codon ``"EM106"`` or a tuple/list
        ``[domain_code, category, concept]`` into a Codon."""
        if isinstance(codon_repr, str):
            try:
                return Codon.from_str(codon_repr)
            except (ValueError, IndexError):
                return None
        if isinstance(codon_repr, (list, tuple)) and len(codon_repr) >= 3:
            d_code = codon_repr[0]
            d_id = DOMAIN_CODES.get(d_code) if isinstance(d_code, str) else d_code
            if d_id is None:
                return None
            return Codon(domain=d_id, category=int(codon_repr[1]), concept=int(codon_repr[2]))
        return None

    def _entry_from_raw(self, word: str, raw: dict) -> CodebookEntry | None:
        domain_code = raw["d"]
        domain_id = DOMAIN_CODES.get(domain_code)
        if domain_id is None:
            return None
        codon = Codon(domain=domain_id, category=raw["c"], concept=raw["n"])

        related: list[tuple[Codon, int]] = []
        for rel in raw.get("rel", []):
            try:
                rel_codon_repr, rel_w = rel[0], int(rel[1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            rel_codon = self._decode_codon(rel_codon_repr)
            if rel_codon is None or rel_codon.is_null:
                continue
            related.append((rel_codon, max(0, min(255, rel_w))))

        return CodebookEntry(
            word=word,
            codon=codon,
            domain_code=domain_code,
            shade_hint=raw.get("s", {}),
            related=tuple(related),
            synset=raw.get("syn", ""),
        )

    def lookup(self, word: str, *, mode: str = "text") -> CodebookEntry | None:
        """Look up a word.

        ``mode``:
          - ``"text"`` (default): text entries only.
          - ``"code"``: code entries only.
          - ``"auto"``: try code first, then text. Used by code encoder for
            structural keywords; identifiers fall through to text.
        """
        w = word.lower()
        if mode == "code":
            raw = self._code_entries.get(w)
        elif mode == "auto":
            raw = self._code_entries.get(w) or self._entries.get(w)
        else:
            raw = self._entries.get(w)
        if raw is None:
            return None
        return self._entry_from_raw(w, raw)

    def synonyms_of(self, word: str) -> list[str]:
        entry = self.lookup(word)
        if entry is None:
            return []
        target = (entry.domain_code, entry.codon.category, entry.codon.concept)
        return [
            w
            for w, raw in self._entries.items()
            if (raw["d"], raw["c"], raw["n"]) == target and w != word.lower()
        ]

    @classmethod
    def load(cls, path: Path | str) -> Codebook:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return cls(json.load(f), base_dir=path.parent)

    def merge_extension(self, extension: dict) -> None:
        """Merge an extension dict (spec §14.3 format) into this codebook.

        Extension entries override base entries. Code entries also merge.
        Returns nothing — modifies in place.
        """
        ext_entries = extension.get("entries", {})
        for word, raw in ext_entries.items():
            self._entries[word.lower()] = raw
        ext_code = extension.get("code_entries", {})
        for word, raw in ext_code.items():
            self._code_entries[word.lower()] = raw

    @classmethod
    def load_with_extensions(cls, base_path: Path | str,
                             extensions: list[Path | str]) -> Codebook:
        """Load a base codebook and merge a list of extensions in order."""
        cb = cls.load(base_path)
        for ext_path in extensions:
            with Path(ext_path).open("r", encoding="utf-8") as f:
                cb.merge_extension(json.load(f))
        return cb


_DEFAULT_CODEBOOK_PATH = Path(__file__).parent / "data" / "codebook_v0.1.0.json"
_DEFAULT_PATCHES_PATH = Path(__file__).parent / "data" / "codebook.patches.json"
_DEFAULT_EXTENSIONS_DIR = Path(__file__).parent / "data" / "extensions"
_default_cache: Codebook | None = None


def default_codebook(*, apply_patches: bool = True) -> Codebook:
    """Load the default codebook with patch + extension overlays.

    Layered loading:
      1. Base codebook JSON (everything from the build pipeline).
      2. Optional patches file (``codebook.patches.json``).
      3. Optional auto-loaded extensions in ``data/extensions/auto/``.

    Patch entries use the regular codebook-entry schema.
    """
    global _default_cache
    if _default_cache is None:
        if not _DEFAULT_CODEBOOK_PATH.exists():
            raise FileNotFoundError(
                f"Default codebook not found at {_DEFAULT_CODEBOOK_PATH}. "
                f"Run `strand build-codebook` first."
            )
        _default_cache = Codebook.load(_DEFAULT_CODEBOOK_PATH)

        if apply_patches and _DEFAULT_PATCHES_PATH.exists():
            try:
                with _DEFAULT_PATCHES_PATH.open("r", encoding="utf-8") as f:
                    _default_cache.merge_extension(json.load(f))
            except (OSError, json.JSONDecodeError):
                pass

        auto_dir = _DEFAULT_EXTENSIONS_DIR / "auto"
        if apply_patches and auto_dir.is_dir():
            for ext_path in sorted(auto_dir.glob("*.json")):
                try:
                    with ext_path.open("r", encoding="utf-8") as f:
                        _default_cache.merge_extension(json.load(f))
                except (OSError, json.JSONDecodeError):
                    continue
    return _default_cache


def reset_default_codebook_cache() -> None:
    """Force a re-read of the default codebook on next access."""
    global _default_cache
    _default_cache = None
