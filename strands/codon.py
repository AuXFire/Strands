from __future__ import annotations

import struct
from dataclasses import dataclass

DOMAIN_CODES: dict[str, int] = {
    "EM": 0x00, "AC": 0x01, "OB": 0x02, "QU": 0x03, "AB": 0x04,
    "NA": 0x05, "PE": 0x06, "SP": 0x07, "TM": 0x08, "QT": 0x09,
    "BD": 0x0A, "SO": 0x0B, "TC": 0x0C, "FD": 0x0D, "CM": 0x0E,
    "SN": 0x0F, "MV": 0x10, "RL": 0x11, "EC": 0x12,
    # Code domains (Phase 2 — no entries yet)
    "CF": 0x20, "DS": 0x21, "TS": 0x22, "OP": 0x23, "IO": 0x24,
    "ER": 0x25, "PT": 0x26, "MD": 0x27, "TE": 0x28, "AP": 0x29, "IN": 0x2A,
    # Structured data domains
    "SC": 0x40, "TR": 0x41, "FM": 0x42,
}

DOMAIN_NAMES: dict[str, str] = {
    "EM": "Emotion", "AC": "Action", "OB": "Object", "QU": "Quality",
    "AB": "Abstract", "NA": "Nature", "PE": "Person", "SP": "Space",
    "TM": "Time", "QT": "Quantity", "BD": "Body", "SO": "Social",
    "TC": "Tech", "FD": "Food", "CM": "Communication", "SN": "Sensory",
    "MV": "Movement", "RL": "Relation", "EC": "Economy",
    "CF": "Control Flow", "DS": "Data Structure", "TS": "Type System",
    "OP": "Operation", "IO": "IO", "ER": "Error", "PT": "Pattern",
    "MD": "Module", "TE": "Testing", "AP": "API", "IN": "Infrastructure",
    "SC": "Schema", "TR": "Transform", "FM": "Format",
}

_ID_TO_CODE: dict[int, str] = {v: k for k, v in DOMAIN_CODES.items()}


@dataclass(frozen=True, slots=True)
class Codon:
    domain: int
    category: int
    concept: int

    def to_bytes(self) -> bytes:
        return struct.pack("BBB", self.domain, self.category, self.concept)

    @classmethod
    def from_bytes(cls, data: bytes) -> Codon:
        d, c, n = struct.unpack("BBB", data[:3])
        return cls(domain=d, category=c, concept=n)

    @property
    def domain_code(self) -> str:
        return _ID_TO_CODE.get(self.domain, "??")

    @property
    def is_null(self) -> bool:
        """A null codon has domain=0xFF — sentinel for unused related-codon
        slots in v2 strands. All real domain IDs are <= 0x42."""
        return self.domain == 0xFF

    def to_str(self) -> str:
        return f"{self.domain_code}{self.category:01X}{self.concept:02X}"

    @classmethod
    def from_str(cls, s: str) -> Codon:
        code = s[:2].upper()
        if code not in DOMAIN_CODES:
            raise ValueError(f"Unknown domain code: {code}")
        domain = DOMAIN_CODES[code]
        category = int(s[2], 16)
        concept = int(s[3:5], 16)
        return cls(domain=domain, category=category, concept=concept)


# Sentinel codon used to fill empty related-codon slots in v2 strands.
# Domain 0xFF is reserved and never assigned to real entries.
NULL_CODON = Codon(domain=0xFF, category=0xFF, concept=0xFF)
