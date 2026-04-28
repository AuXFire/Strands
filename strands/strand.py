from __future__ import annotations

import struct
from dataclasses import dataclass, field

from strands.codon import Codon
from strands.shade import Shade

MAGIC = b"SS"
VERSION = 0x01


@dataclass(frozen=True, slots=True)
class CodonEntry:
    codon: Codon
    shade: int
    word: str = ""
    alt_codons: tuple[Codon, ...] = ()
    synset: str = ""

    @property
    def shade_obj(self) -> Shade:
        return Shade.from_byte(self.shade)


@dataclass(slots=True)
class Strand:
    codons: list[CodonEntry] = field(default_factory=list)

    @property
    def byte_size(self) -> int:
        return 8 + 4 * len(self.codons)

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

    def to_binary(self, metadata: bytes | None = None, code_mode: bool = False) -> bytes:
        flags = 0
        if metadata:
            flags |= 0x01
        if code_mode:
            flags |= 0x02

        header = MAGIC + struct.pack(">BBHxx", VERSION, flags, len(self.codons))
        body = b"".join(
            struct.pack("BBBB", e.codon.domain, e.codon.category, e.codon.concept, e.shade)
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
        if version != VERSION:
            raise ValueError(f"Unsupported strand version: {version}")

        entries = []
        offset = 8
        for _ in range(count):
            d, c, n, s = struct.unpack("BBBB", data[offset : offset + 4])
            entries.append(CodonEntry(codon=Codon(d, c, n), shade=s))
            offset += 4
        return cls(codons=entries)

    @property
    def domains(self) -> set[int]:
        return {e.codon.domain for e in self.codons}
