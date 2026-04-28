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


class Codebook:
    def __init__(self, data: dict):
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

    def _entry_from_raw(self, word: str, raw: dict) -> CodebookEntry | None:
        domain_code = raw["d"]
        domain_id = DOMAIN_CODES.get(domain_code)
        if domain_id is None:
            return None
        codon = Codon(domain=domain_id, category=raw["c"], concept=raw["n"])
        return CodebookEntry(
            word=word,
            codon=codon,
            domain_code=domain_code,
            shade_hint=raw.get("s", {}),
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
