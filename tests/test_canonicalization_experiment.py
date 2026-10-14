"""Unit tests for transaction canonicalization experiment components."""

import pytest
import numpy as np
from src.canonicalize import canonicalize, canonicalize_series
from scripts.run_canonicalization_experiment import extract_handcrafted_features


def test_canonicalize_chain_and_whitespace_preservation():
    """Verify canonicalizer preserves merchant identity across complex wrappers."""
    raw = "UPI/99887766/UBER INDIA MUMBAI/uber@okaxis"
    assert canonicalize(raw) == "UBER INDIA MUMBAI"

    # Preserves normal uncorrupted strings without alteration
    clean = "Swiggy Delivery Koramangala"
    assert canonicalize(clean) == clean


def test_canonicalize_empty_string_safety():
    """Verify fallback behavior when text is purely punctuation or numbers."""
    # Stripping long numbers should leave non-empty original if all tokens removed
    assert canonicalize("123456789012") != ""
    assert canonicalize("   ") == ""
    assert canonicalize(None) == ""


def test_extract_handcrafted_features_shape_and_values():
    """Verify morphological feature extractor dimensions and bounds."""
    texts = [
        "UPI/123/SWIGGY/swiggy@axis",
        "NEFT-999-RENT",
        "STARBUCKS COFFEE MUMBAI"
    ]
    feats = extract_handcrafted_features(texts)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (3, 7)

    # Check UPI flag
    assert feats[0, 3] == 1.0  # has_upi
    assert feats[0, 4] == 0.0  # has_neft

    # Check NEFT flag
    assert feats[1, 3] == 0.0
    assert feats[1, 4] == 1.0  # has_neft

    # Check ratios are within [0, 1]
    for row in feats:
        assert 0.0 <= row[1] <= 1.0  # digit ratio
        assert 0.0 <= row[2] <= 1.0  # upper ratio
