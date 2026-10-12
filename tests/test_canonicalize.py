import pytest
from src.canonicalize import canonicalize, canonicalize_series

def test_canonicalize_upi():
    raw = "UPI/4829103948/SWIGGY/swiggy@okaxis"
    assert canonicalize(raw) == "SWIGGY"

def test_canonicalize_neft():
    raw = "NEFT-92837492-AMAZON RETAIL"
    assert canonicalize(raw) == "AMAZON RETAIL"

def test_canonicalize_imps():
    raw = "IMPS/83910283/NETFLIX INDIA/REF991"
    assert canonicalize(raw) == "NETFLIX INDIA"

def test_canonicalize_pos():
    raw = "POS 4129XXXX9812 RELIANCE RETAIL MUMBAI"
    assert canonicalize(raw) == "RELIANCE RETAIL"

def test_canonicalize_masked_cards():
    raw = "Card Txn 5412XXXX3910 Uber Trips"
    assert canonicalize(raw) == "Card Txn Uber Trips"

def test_canonicalize_series_raw_flag():
    texts = [
        "UPI/11223344/STARBUCKS/sb@okaxis",
        "NEFT-556677-RENT"
    ]
    # When raw=False, canonicalize
    clean = canonicalize_series(texts, raw=False)
    assert clean == ["STARBUCKS", "RENT"]

    # When raw=True, preserve original
    raw_out = canonicalize_series(texts, raw=True)
    assert raw_out == texts

def test_canonicalize_empty_and_edge_cases():
    assert canonicalize("") == ""
    assert canonicalize("   ") == ""
    assert canonicalize("Zomato") == "Zomato"
    assert canonicalize("Amazon India Marketplace") == "Amazon India Marketplace"
    # Non-string input handling
    assert canonicalize(None) == ""
    assert canonicalize(12345) == ""
    assert canonicalize(float("nan")) == ""
