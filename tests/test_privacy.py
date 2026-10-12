import pytest
from src.privacy import PrivacyGuard, PrivacyViolationError

def test_validate_allowed_payload():
    valid_payload = {
        "status": "success",
        "intent": "spending_breakdown",
        "total_spending": 24500.0,
        "category_spending": {"Food & Dining": 12000.0, "Shopping & Retail": 12500.0},
        "currency": "INR"
    }
    assert PrivacyGuard.validate_outbound_payload(valid_payload) is True

def test_reject_raw_transactions():
    bad_payload = {
        "status": "success",
        "raw_transactions": [
            {"date": "2026-08-01", "transaction_description": "Private sensitive purchase", "amount": 100.0}
        ]
    }
    with pytest.raises(PrivacyViolationError, match="Data Minimization Violation"):
        PrivacyGuard.validate_outbound_payload(bad_payload)

def test_reject_unapproved_field():
    bad_payload = {
        "status": "success",
        "user_secret_notes": "Internal user notes"
    }
    with pytest.raises(PrivacyViolationError, match="not on the approved outbound allowlist"):
        PrivacyGuard.validate_outbound_payload(bad_payload)

def test_sanitize_payload_redacts_pii():
    payload = {
        "summary": "Payment processed for card 4111 2222 3333 4444 at Store",
        "total_spending": 5000.0,
        "unapproved_field": "Should be dropped"
    }
    sanitized = PrivacyGuard.sanitize_outbound_payload(payload)
    assert "[REDACTED_PII]" in sanitized["summary"]
    assert "4111" not in sanitized["summary"]
    assert "unapproved_field" not in sanitized
    assert sanitized["total_spending"] == 5000.0
