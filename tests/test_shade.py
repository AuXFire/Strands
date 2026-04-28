from strands.shade import Shade, compute_shade, shade_similarity


def test_shade_byte_round_trip():
    s = Shade(intensity=2, abstraction=1, formality=3, polarity=0)
    b = s.to_byte()
    assert Shade.from_byte(b) == s


def test_shade_bit_layout():
    s = Shade(intensity=3, abstraction=0, formality=0, polarity=0)
    assert s.to_byte() == 0b11_00_00_00


def test_shade_similarity_extremes():
    assert shade_similarity(0, 0) == 1.0
    assert shade_similarity(0, 255) < 0.01


def test_compute_shade_uses_hint():
    b = compute_shade("happy", {"p": 3, "f": 1, "i": 2})
    s = Shade.from_byte(b)
    assert s.polarity == 3
    assert s.intensity == 2
    assert s.formality == 1
