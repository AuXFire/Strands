"""In-memory index with brute-force search and domain pre-filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strands.codebook import Codebook
from strands.comparator import ComparisonResult, compare_strands
from strands.encoder import encode
from strands.strand import Strand


@dataclass(slots=True)
class IndexEntry:
    id: str
    content: str
    strand: Strand
    domains: frozenset[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    entry: IndexEntry
    score: float
    comparison: ComparisonResult


class InMemoryIndex:
    def __init__(self, codebook: Codebook | None = None):
        self._entries: list[IndexEntry] = []
        self._codebook = codebook

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, id: str, content: str, metadata: dict[str, Any] | None = None) -> IndexEntry:
        result = encode(content, codebook=self._codebook)
        entry = IndexEntry(
            id=id,
            content=content,
            strand=result.strand,
            domains=frozenset(result.strand.domains),
            metadata=metadata or {},
        )
        self._entries.append(entry)
        return entry

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        threshold: float = 0.0,
        domains: list[int] | None = None,
    ) -> list[SearchResult]:
        query_result = encode(query, codebook=self._codebook)
        query_strand = query_result.strand
        query_domains = frozenset(query_strand.domains)

        if domains is not None:
            domain_filter = frozenset(domains)
            candidates = [e for e in self._entries if e.domains & domain_filter]
        elif query_domains:
            candidates = [e for e in self._entries if e.domains & query_domains]
        else:
            candidates = list(self._entries)

        results: list[SearchResult] = []
        for entry in candidates:
            comparison = compare_strands(query_strand, entry.strand)
            if comparison.score >= threshold:
                results.append(
                    SearchResult(entry=entry, score=comparison.score, comparison=comparison)
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
