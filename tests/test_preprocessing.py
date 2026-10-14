import pytest
from src.preprocessing import TextPreprocessor

def test_text_cleaning_lowercasing():
    prep = TextPreprocessor(lowercase=True)
    res = prep.clean_text("AMAZON STORE #1029")
    assert res == "amazon store #1029"

def test_text_cleaning_whitespace():
    prep = TextPreprocessor(normalize_whitespace=True)
    res = prep.clean_text("  Uber   Ride   TXN123  ")
    assert res == "uber ride txn123"

def test_text_cleaning_urls():
    prep = TextPreprocessor()
    res = prep.clean_text("Payment at https://www.netflix.com/sub")
    assert "https" not in res
    assert "www" not in res
    assert res == "payment at"

def test_text_cleaning_empty_input():
    prep = TextPreprocessor()
    assert prep.clean_text("") == ""
    assert prep.clean_text(None) == ""


def test_preprocessing_upi_wrapper():
    prep = TextPreprocessor()
    assert prep.clean_text("UPI/412398124/SWIGGY/swiggy@icici") == "swiggy"


def test_preprocessing_neft_wrapper():
    prep = TextPreprocessor()
    assert prep.clean_text("NEFT-981241-NETFLIX") == "netflix"


def test_preprocessing_imps_wrapper():
    prep = TextPreprocessor()
    assert prep.clean_text("IMPS/981234123/UBER/REF12345") == "uber"


def test_preprocessing_pos_wrapper():
    prep = TextPreprocessor()
    assert prep.clean_text("POS 4129XXXX9812 RELIANCE FRESH MUMBAI") == "reliance fresh"


def test_preprocessing_whitespace_and_empty_edge_cases():
    import pandas as pd
    prep = TextPreprocessor()
    assert prep.clean_text("   ") == ""
    assert prep.clean_text("\t\n  \r  ") == ""
    assert prep.clean_text(pd.NA) == ""
    assert prep.clean_text(float("nan")) == ""


def test_preprocessing_ordinary_merchant_names():
    prep = TextPreprocessor()
    assert prep.clean_text("Starbucks Coffee") == "starbucks coffee"
    assert prep.clean_text("Amazon Retail Store #1029") == "amazon retail store #1029"
    assert prep.clean_text("Uber Trip") == "uber trip"


def test_preprocessing_idempotence():
    prep = TextPreprocessor()
    test_samples = [
        "UPI/412398124/SWIGGY/swiggy@icici",
        "NEFT-981241-NETFLIX",
        "IMPS/981234123/UBER/REF12345",
        "POS 4129XXXX9812 RELIANCE FRESH MUMBAI",
        "Starbucks Coffee",
        "  Multiple   Spaces   In   Name  ",
        ""
    ]
    for text in test_samples:
        first_pass = prep.clean_text(text)
        second_pass = prep.clean_text(first_pass)
        assert first_pass == second_pass, f"Failed idempotence on {text}: {first_pass} != {second_pass}"


def test_preprocessing_disable_canonicalization_flag():
    prep_no_canon = TextPreprocessor(use_canonicalization=False)
    cleaned = prep_no_canon.clean_text("UPI/412398124/SWIGGY/swiggy@icici")
    assert "upi" in cleaned
    assert "swiggy" in cleaned
    assert "icici" in cleaned
