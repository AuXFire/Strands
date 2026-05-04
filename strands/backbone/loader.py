"""Memory-mapped backbone loader.

Loading is O(1) — the binary tables become numpy memmap arrays, accessed
in-place from the OS page cache. No JSON parsing per node, no Python
object overhead per edge. Useful for ~200 MB backbones running on
consumer laptops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from strands.backbone.schema import EDGE_DTYPE, NODE_DTYPE, Rel


@dataclass(slots=True)
class BackboneNode:
    """Lightweight wrapper around one row of the node array."""
    node_id: int
    concept_type: int
    activation_default: int
    lemmas: tuple[str, ...]
    codon: tuple[int, int, int] | None  # (domain, category, concept) or None
    relationship_count: int
    relationship_offset: int


class Backbone:
    """In-memory view of a compiled backbone."""

    def __init__(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        lemmas_buf: bytes,
        manifest: dict,
        glosses_buf: bytes = b"\x00",
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.lemmas_buf = lemmas_buf
        self.glosses_buf = glosses_buf
        self.manifest = manifest
        # Built lazily.
        self._lemma_to_node_ids: dict[str, list[int]] | None = None

    # ---- Counts -----------------------------------------------------------

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edges.shape[0])

    # ---- Lookups ----------------------------------------------------------

    def lemmas_for(self, node_id: int) -> tuple[str, ...]:
        n = self.nodes[node_id]
        count = int(n["lemma_count"])
        offset = int(n["lemma_offset"])
        if count == 0:
            return ()
        result: list[str] = []
        cursor = offset
        for _ in range(count):
            end = self.lemmas_buf.find(b"\x00", cursor)
            if end == -1:
                break
            result.append(self.lemmas_buf[cursor:end].decode("utf-8"))
            cursor = end + 1
        return tuple(result)

    def edges_of(self, node_id: int) -> np.ndarray:
        """View into the edges array for ``node_id``'s outgoing edges.
        Read-only; modifying the returned slice mutates the mmap."""
        n = self.nodes[node_id]
        offset = int(n["relationship_offset"])
        count = int(n["relationship_count"])
        return self.edges[offset : offset + count]

    def codon_of(self, node_id: int) -> tuple[int, int, int] | None:
        n = self.nodes[node_id]
        d = int(n["codon_domain"])
        if d == 0xFF:
            return None
        return (d, int(n["codon_category"]), int(n["codon_concept"]))

    def gloss_for(self, node_id: int) -> str:
        """Return the WordNet definition of ``node_id`` or empty string."""
        if "gloss_offset" not in self.nodes.dtype.names:
            return ""
        offset = int(self.nodes[node_id]["gloss_offset"])
        if offset == 0:
            return ""
        end = self.glosses_buf.find(b"\x00", offset)
        if end == -1:
            return ""
        return self.glosses_buf[offset:end].decode("utf-8", errors="replace")

    def node(self, node_id: int) -> BackboneNode:
        n = self.nodes[node_id]
        return BackboneNode(
            node_id=int(n["node_id"]),
            concept_type=int(n["concept_type"]),
            activation_default=int(n["activation_default"]),
            lemmas=self.lemmas_for(node_id),
            codon=self.codon_of(node_id),
            relationship_count=int(n["relationship_count"]),
            relationship_offset=int(n["relationship_offset"]),
        )

    # ---- Reverse lookup: lemma → nodes -----------------------------------

    def _build_lemma_index(self) -> None:
        idx: dict[str, list[int]] = {}
        for i in range(self.node_count):
            for lemma in self.lemmas_for(i):
                idx.setdefault(lemma, []).append(i)
        self._lemma_to_node_ids = idx

    def nodes_for_lemma(self, lemma: str) -> list[int]:
        if self._lemma_to_node_ids is None:
            self._build_lemma_index()
        return list(self._lemma_to_node_ids.get(lemma.lower(), ()))

    # ---- Edge filtering ---------------------------------------------------

    def edges_with_relation(
        self, node_id: int, rel: int | Rel,
    ) -> np.ndarray:
        edges = self.edges_of(node_id)
        if edges.size == 0:
            return edges
        rel_id = int(rel)
        mask = edges["relation_type"] == rel_id
        return edges[mask]


def load(out_dir: Path | str) -> Backbone:
    """Load a backbone from a directory containing the four artifact files."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "backbone.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No backbone manifest at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    nodes_path = out_dir / manifest["files"]["nodes"]
    edges_path = out_dir / manifest["files"]["edges"]
    lemmas_path = out_dir / manifest["files"]["lemmas"]
    glosses_filename = manifest["files"].get("glosses")

    # Memory-map for O(1) loading regardless of file size.
    nodes = np.memmap(nodes_path, dtype=NODE_DTYPE, mode="r")
    edges = np.memmap(edges_path, dtype=EDGE_DTYPE, mode="r")
    lemmas_buf = lemmas_path.read_bytes()
    glosses_buf = b"\x00"
    if glosses_filename:
        glosses_path = out_dir / glosses_filename
        if glosses_path.exists():
            glosses_buf = glosses_path.read_bytes()

    if int(manifest["node_count"]) != nodes.shape[0]:
        raise ValueError(
            f"Node count mismatch: manifest says {manifest['node_count']}, "
            f"file has {nodes.shape[0]}"
        )
    if int(manifest["edge_count"]) != edges.shape[0]:
        raise ValueError(
            f"Edge count mismatch: manifest says {manifest['edge_count']}, "
            f"file has {edges.shape[0]}"
        )

    return Backbone(
        nodes=nodes, edges=edges, lemmas_buf=lemmas_buf,
        manifest=manifest, glosses_buf=glosses_buf,
    )
