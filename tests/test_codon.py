from strands.codon import DOMAIN_CODES, Codon


def test_codon_to_str_round_trip():
    c = Codon(domain=DOMAIN_CODES["EM"], category=1, concept=6)
    s = c.to_str()
    assert s == "EM106"
    assert Codon.from_str(s) == c


def test_codon_bytes_round_trip():
    c = Codon(domain=0x05, category=0x0A, concept=0xFF)
    raw = c.to_bytes()
    assert len(raw) == 3
    assert Codon.from_bytes(raw) == c


def test_domain_table_has_all_text_domains():
    expected = {
        "EM", "AC", "OB", "QU", "AB", "NA", "PE", "SP", "TM", "QT",
        "BD", "SO", "TC", "FD", "CM", "SN", "MV", "RL", "EC",
    }
    assert expected.issubset(DOMAIN_CODES.keys())
