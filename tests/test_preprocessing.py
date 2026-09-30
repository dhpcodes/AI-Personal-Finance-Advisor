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
