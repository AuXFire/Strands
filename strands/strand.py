"""Strand: ordered sequence of (codon + shade [+ relations]) entries.

Two binary formats:

  v1 — 4 bytes per token. Spec-compatible.
    [0-2] Domain | Category | Concept       (3 bytes)
    [3]   Shade                              (1 byte)

  v2 — 20 bytes per token. Adds four ConceptNet-derived related-codon
       slots. Self-contained: at compare time, two strands need only
       their own bytes — no codebook, no runtime model, no sidecar
       files.
    [0-2]  Primary codon                     (3 bytes)
    [3]    Shade                             (1 byte)
    [4-6]  Related codon #1                  (3 bytes)
    [7]    Related codon #1 weight (u8)      (1 byte)
    [8-10] Related codon #2                  (3 bytes)
    [11]   Related codon #2 weight (u8)      (1 byte)
    [12-14] Related codon #3                 (3 bytes)
    [15]    Related codon #3 weight (u8)     (1 byte)
    [16-18] Related codon #4                 (3 bytes)
    [19]    Related codon #4 weight (u8)     (1 byte)

  Empty related slots are filled with NULL_CODON (domain 0xFF) and
  weight 0. The header has a flag bit; readers auto-detect format.
  Storage is 60× smaller than GloVe-300 (1200 bytes) and 307× smaller
  than OpenAI text-embedding-3-small (6144 bytes) per token.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

V2_RELATED_SLOTS = 4

from strands.codon import NULL_CODON, Codon
from strands.shade import Shade

MAGIC = b"SS"
VERSION_V1 = 0x01
VERSION_V2 = 0x02
FLAG_HAS_METADATA = 0x01
FLAG_CODE_MODE = 0x02
FLAG_V2_RELATIONS = 0x04  # 12-byte body entries with two related codons


@dataclass(frozen=True, slots=True)
class CodonEntry:
    codon: Codon
    shade: int
    word: str = ""
    # Up to two ConceptNet-derived related codons with their uint8 weights.
    # Stamped in by the encoder from the codebook entry. Used by the
    # comparator for in-strand cross-codon relatedness scoring.
    related: tuple[tuple[Codon, int], ...] = ()
    # Optional metadata for round-trip diagnostics; not in binary format.
    synset: str = ""

    @property
    def shade_obj(self) -> Shade:
        return Shade.from_byte(self.shade)


@dataclass(slots=True)
class Strand:
    codons: list[CodonEntry] = field(default_factory=list)
    version: int = VERSION_V2  # default to v2 going forward

    @property
    def byte_size(self) -> int:
        """Binary byte size for the strand's current version."""
        if self.version == VERSION_V2:
            per_token = 4 + 4 * V2_RELATED_SLOTS
        else:
            per_token = 4
        return 8 + per_token * len(self.codons)

    def to_text(self) -> str:
        """Human-readable text form. Unchanged across versions —
        debugging-only, doesn't carry related codons."""
        return "·".join(f"{e.codon.to_str()}:{e.shade:02X}" for e in self.codons)

    @classmethod
    def from_text(cls, text: str) -> Strand:
        entries = []
        for part in text.split("·"):
            part = part.strip()
            if not part:
                continue
            codon_s, shade_s = part.split(":")
            codon = Codon.from_str(codon_s)
            shade = int(shade_s, 16)
            entries.append(CodonEntry(codon=codon, shade=shade))
        return cls(codons=entries, version=VERSION_V1)

    def to_binary(
        self,
        metadata: bytes | None = None,
        code_mode: bool = False,
        *,
        version: int | None = None,
    ) -> bytes:
        """Serialize to binary.

        ``version`` overrides the strand's stored version (default uses
        ``self.version``). v2 emits 12-byte token entries with related
        codons; v1 emits 4-byte spec-compatible entries.
        """
        v = version if version is not None else self.version
        flags = 0
        if metadata:
            flags |= FLAG_HAS_METADATA
        if code_mode:
            flags |= FLAG_CODE_MODE
        if v == VERSION_V2:
            flags |= FLAG_V2_RELATIONS

        header = MAGIC + struct.pack(">BBHxx", v, flags, len(self.codons))

        if v == VERSION_V2:
            body = bytearray()
            for e in self.codons:
                body.extend(struct.pack(
                    "BBBB",
                    e.codon.domain & 0xFF,
                    e.codon.category & 0xFF,
                    e.codon.concept & 0xFF,
                    e.shade & 0xFF,
                ))
                # Pad related to V2_RELATED_SLOTS with NULL_CODON.
                rels = list(e.related[:V2_RELATED_SLOTS])
                while len(rels) < V2_RELATED_SLOTS:
                    rels.append((NULL_CODON, 0))
                for rel_codon, rel_w in rels:
                    body.extend(struct.pack(
                        "BBBB",
                        rel_codon.domain & 0xFF,
                        rel_codon.category & 0xFF,
                        rel_codon.concept & 0xFF,
                        rel_w & 0xFF,
                    ))
            body_bytes = bytes(body)
        else:
            body_bytes = b"".join(
                struct.pack(
                    "BBBB",
                    e.codon.domain & 0xFF,
                    e.codon.category & 0xFF,
                    e.codon.concept & 0xFF,
                    e.shade & 0xFF,
                )
                for e in self.codons
            )

        result = header + body_bytes
        if metadata:
            result += struct.pack(">H", len(metadata)) + metadata
        return result

    @classmethod
    def from_binary(cls, data: bytes) -> Strand:
        if data[:2] != MAGIC:
            raise ValueError("Invalid strand binary: bad magic")
        version, flags, count = struct.unpack(">BBH", data[2:6])
        if version not in (VERSION_V1, VERSION_V2):
            raise ValueError(f"Unsupported strand version: {version}")

        is_v2 = bool(flags & FLAG_V2_RELATIONS) or version == VERSION_V2
        entries = []
        offset = 8

        if is_v2:
            for _ in range(count):
                d, c, n, s = struct.unpack(
                    "BBBB", data[offset : offset + 4]
                )
                offset += 4
                rels = []
                for _ in range(V2_RELATED_SLOTS):
                    rd, rc, rn, rw = struct.unpack(
                        "BBBB", data[offset : offset + 4]
                    )
                    offset += 4
                    rel_codon = Codon(rd, rc, rn)
                    if not rel_codon.is_null:
                        rels.append((rel_codon, rw))
                entries.append(CodonEntry(
                    codon=Codon(d, c, n), shade=s,
                    related=tuple(rels),
                ))
        else:
            for _ in range(count):
                d, c, n, s = struct.unpack("BBBB", data[offset : offset + 4])
                entries.append(CodonEntry(codon=Codon(d, c, n), shade=s))
                offset += 4

        return cls(codons=entries, version=VERSION_V2 if is_v2 else VERSION_V1)

    @property
    def domains(self) -> set[int]:
        return {e.codon.domain for e in self.codons}
