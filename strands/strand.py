from __future__ import annotations

import struct
from dataclasses import dataclass, field

from strands.codon import Codon
from strands.shade import Shade

MAGIC = b"SS"
VERSION = 0x01
VERSION_V2 = 0x02
FLAG_HAS_METADATA = 0x01
FLAG_CODE_MODE = 0x02
FLAG_V2_EXTENDED = 0x04  # 8-byte body entries instead of 4-byte


@dataclass(frozen=True, slots=True)
class CodonEntry:
    codon: Codon
    shade: int
    word: str = ""
    alt_codons: tuple[Codon, ...] = ()
    synset: str = ""
    sense_rank: int = 0           # 0 = primary; >0 = which alt was chosen
    semantic_role: int = 0        # 0 = unspecified; per spec §3.1 v2 extension
    source_position: int = 0      # token index in source (uint16)

    @property
    def shade_obj(self) -> Shade:
        return Shade.from_byte(self.shade)


@dataclass(slots=True)
class Strand:
    codons: list[CodonEntry] = field(default_factory=list)

    @property
    def byte_size(self) -> int:
        """v1 byte size (default). For v2 extended, multiply by 2x."""
        return 8 + 4 * len(self.codons)

    @property
    def byte_size_v2(self) -> int:
        return 8 + 8 * len(self.codons)

    def to_text(self) -> str:
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
        return cls(codons=entries)

    def to_binary(
        self,
        metadata: bytes | None = None,
        code_mode: bool = False,
        *,
        extended: bool = False,
    ) -> bytes:
        """Serialize to binary. ``extended=True`` emits Strand v2 with
        8 bytes per codon (sense_rank, semantic_role, source_position
        added). Default 4-byte v1 format is unchanged."""
        flags = 0
        if metadata:
            flags |= FLAG_HAS_METADATA
        if code_mode:
            flags |= FLAG_CODE_MODE
        if extended:
            flags |= FLAG_V2_EXTENDED
            version = VERSION_V2
        else:
            version = VERSION

        header = MAGIC + struct.pack(">BBHxx", version, flags, len(self.codons))
        if extended:
            body = b"".join(
                struct.pack(
                    ">BBBBBBH",
                    e.codon.domain, e.codon.category, e.codon.concept, e.shade,
                    e.sense_rank & 0xFF,
                    e.semantic_role & 0xFF,
                    e.source_position & 0xFFFF,
                )
                for e in self.codons
            )
        else:
            body = b"".join(
                struct.pack(
                    "BBBB",
                    e.codon.domain, e.codon.category, e.codon.concept, e.shade,
                )
                for e in self.codons
            )
        result = header + body
        if metadata:
            result += struct.pack(">H", len(metadata)) + metadata
        return result

    @classmethod
    def from_binary(cls, data: bytes) -> Strand:
        if data[:2] != MAGIC:
            raise ValueError("Invalid strand binary: bad magic")
        version, flags, count = struct.unpack(">BBH", data[2:6])
        if version not in (VERSION, VERSION_V2):
            raise ValueError(f"Unsupported strand version: {version}")

        extended = bool(flags & FLAG_V2_EXTENDED)
        entries = []
        offset = 8
        if extended:
            for _ in range(count):
                d, c, n, s, sr, sem, pos = struct.unpack(
                    ">BBBBBBH", data[offset : offset + 8]
                )
                entries.append(CodonEntry(
                    codon=Codon(d, c, n), shade=s,
                    sense_rank=sr, semantic_role=sem, source_position=pos,
                ))
                offset += 8
        else:
            for _ in range(count):
                d, c, n, s = struct.unpack("BBBB", data[offset : offset + 4])
                entries.append(CodonEntry(codon=Codon(d, c, n), shade=s))
                offset += 4
        return cls(codons=entries)

    @property
    def domains(self) -> set[int]:
        return {e.codon.domain for e in self.codons}
