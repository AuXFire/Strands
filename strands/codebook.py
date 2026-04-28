"""Codebook loader — O(1) word → codon lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from strands.codon import DOMAIN_CODES, Codon


@dataclass(frozen=True, slots=True)
class CodebookEntry:
    word: str
    codon: Codon
    domain_code: str
    shade_hint: dict
    alt_codons: tuple[Codon, ...] = ()
    synset: str = ""


class Codebook:
    def __init__(self, data: dict):
        self.version: str = data.get("version", "unknown")
        self.stats: dict = data.get("stats", {})
        self.domains: dict = data.get("domains", {})
        self._entries: dict[str, dict] = data.get("entries", {})
        self._code_entries: dict[str, dict] = data.get("code_entries", {})
        # Codon→codon adjacency for cross-domain relatedness (§6.1).
        # Replaces runtime ConceptNet/Numberbatch lookup with a compact
        # built-in graph: {codon_str: [(neighbor_codon_str, weight_u8), ...]}
        raw_adj: dict = data.get("codon_adjacency", {})
        self._adjacency: dict[str, dict[str, int]] = {
            codon: {nb: w for nb, w in edges} for codon, edges in raw_adj.items()
        }

    @property
    def adjacency_size(self) -> int:
        return sum(len(v) for v in self._adjacency.values())

    def codon_relatedness(self, codon_a_str: str, codon_b_str: str) -> float | None:
        """Return relatedness in [0.0, 1.0] for two codon strings, or None
        if the pair has no edge in the adjacency table. Symmetric: tries
        both directions."""
        edges_a = self._adjacency.get(codon_a_str)
        if edges_a is not None:
            w = edges_a.get(codon_b_str)
            if w is not None:
                return w / 255.0
        edges_b = self._adjacency.get(codon_b_str)
        if edges_b is not None:
            w = edges_b.get(codon_a_str)
            if w is not None:
                return w / 255.0
        return None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def code_size(self) -> int:
        return len(self._code_entries)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._entries

    def _entry_from_raw(self, word: str, raw: dict) -> CodebookEntry | None:
        domain_code = raw["d"]
        domain_id = DOMAIN_CODES.get(domain_code)
        if domain_id is None:
            return None
        codon = Codon(domain=domain_id, category=raw["c"], concept=raw["n"])

        alt_codons: list[Codon] = []
        for alt in raw.get("alt", []):
            try:
                d_code, cat, conc = alt[0], alt[1], alt[2]
            except (KeyError, IndexError, TypeError):
                continue
            d_id = DOMAIN_CODES.get(d_code)
            if d_id is None:
                continue
            alt_codons.append(Codon(domain=d_id, category=cat, concept=conc))

        return CodebookEntry(
            word=word,
            codon=codon,
            domain_code=domain_code,
            shade_hint=raw.get("s", {}),
            alt_codons=tuple(alt_codons),
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
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

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
_default_cache: Codebook | None = None


def default_codebook() -> Codebook:
    global _default_cache
    if _default_cache is None:
        if not _DEFAULT_CODEBOOK_PATH.exists():
            raise FileNotFoundError(
                f"Default codebook not found at {_DEFAULT_CODEBOOK_PATH}. "
                f"Run `strand build-codebook` first."
            )
        _default_cache = Codebook.load(_DEFAULT_CODEBOOK_PATH)
    return _default_cache
