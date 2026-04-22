from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Shade:
    intensity: int  # 0-3
    abstraction: int  # 0-3
    formality: int  # 0-3
    polarity: int  # 0-3

    def to_byte(self) -> int:
        return (self.intensity << 6) | (self.abstraction << 4) | (self.formality << 2) | self.polarity

    @classmethod
    def from_byte(cls, b: int) -> Shade:
        return cls(
            intensity=(b >> 6) & 0x03,
            abstraction=(b >> 4) & 0x03,
            formality=(b >> 2) & 0x03,
            polarity=b & 0x03,
        )

    def to_hex(self) -> str:
        return f"{self.to_byte():02X}"


def shade_similarity(a: int, b: int) -> float:
    return 1.0 - abs(a - b) / 255


def compute_shade(token: str, hint: dict | None = None) -> int:
    if hint:
        polarity = hint.get("p", 1)
        formality = hint.get("f", 1)
        intensity = hint.get("i", 1)
    else:
        polarity = 1
        formality = 1
        intensity = 1

    abstraction = min(3, len(token) // 4)
    shade = Shade(intensity=intensity, abstraction=abstraction, formality=formality, polarity=polarity)
    return shade.to_byte()
