import pytest
from src.numeric_validator import NumericGroundingValidator


def test_validator_exact_grounded_numbers():
    tool_output = {
        "status": "success",
        "total_expense": 45000.0,
        "currency": "INR",
        "highest_spending_category": "Food & Dining",
        "highest_spending_amount": 18250.0
    }
    reply = "In October, your total spending was ₹45,000, with Food & Dining being your highest category at ₹18,250."
    is_valid, reason = NumericGroundingValidator.validate(reply, tool_output)
    assert is_valid, f"Expected valid grounding, but failed: {reason}"
    assert reason is None


def test_validator_percentage_and_rounding():
    tool_output = {
        "status": "success",
        "total_income": 85000.0,
        "total_expense": 51000.0,
        "savings_rate": 0.40,
        "confidence": 0.9877
    }
    # 40% and 99% are derived representations of 0.40 and 0.9877
    reply = "Your total income was ₹85,000 and expenses were ₹51,000, resulting in a 40% savings rate with 99% confidence."
    is_valid, reason = NumericGroundingValidator.validate(reply, tool_output)
    assert is_valid, f"Expected valid rounding/percentage match, but failed: {reason}"


def test_validator_catches_hallucinated_number():
    tool_output = {
        "status": "success",
        "total_expense": 32000.0,
        "currency": "INR"
    }
    # Hallucinated extra amount ₹15,000 not in tool output
    hallucinated_reply = (
        "Your total expense was ₹32,000 this month. You also have an upcoming EMI of ₹15,000."
    )
    is_valid, reason = NumericGroundingValidator.validate(hallucinated_reply, tool_output)
    assert not is_valid, "Validator should have caught the ungrounded number ₹15,000!"
    assert "15000" in reason or "15,000" in reason or "15" in reason


def test_validator_catches_invented_percentage():
    tool_output = {
        "status": "success",
        "category": "Food & Dining",
        "amount": 12000.0,
        "percentage": 25.0
    }
    # LLM hallucinates 55% instead of 25%
    bad_reply = "Food & Dining accounted for 55% of your budget with ₹12,000."
    is_valid, reason = NumericGroundingValidator.validate(bad_reply, tool_output)
    assert not is_valid
    assert "55" in reason


def test_validator_empty_or_text_only_reply():
    tool_output = {"status": "success", "category": "Shopping & Retail"}
    reply = "Your transaction was classified under Shopping & Retail based on merchant patterns."
    is_valid, reason = NumericGroundingValidator.validate(reply, tool_output)
    assert is_valid
    assert reason is None
