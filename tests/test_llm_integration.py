import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.grounded_llm import GroundedFinancialAssistant
from src.chatbot_tools import ChatbotFinancialTools
from src.anomaly_detection import SpendingAnomalyDetector, format_currency_amount


@pytest.fixture
def sample_inr_df():
    data = [
        {"transaction_id": "tx1", "date": "2026-09-01", "transaction_description": "Salary Deposit", "amount": 100000.0, "category": "Income", "currency": "INR"},
        {"transaction_id": "tx2", "date": "2026-09-05", "transaction_description": "House Rent Payment", "amount": -18000.0, "category": "Housing & Utilities", "currency": "INR"},
        {"transaction_id": "tx3", "date": "2026-09-06", "transaction_description": "House Rent Payment", "amount": -18000.0, "category": "Housing & Utilities", "currency": "INR"}, # Separate legitimate payment
        {"transaction_id": "tx4", "date": "2026-09-15", "transaction_description": "Groceries Supermarket", "amount": -4500.0, "category": "Food & Dining", "currency": "INR"},
        {"transaction_id": "tx5", "date": "2026-09-20", "transaction_description": "Electric Utility Bill", "amount": -1200.0, "category": "Utilities & Services", "currency": "INR"},
    ]
    return pd.DataFrame(data)


def test_api_key_missing_runs_offline(sample_inr_df):
    """Req 1 & 15: Missing API key runs in offline template mode without crashing."""
    assistant = GroundedFinancialAssistant(provider="gemini", api_key=None)
    assert assistant.client is None
    res = assistant.process_query("Where am I spending the most money?", sample_inr_df)
    assert res["is_llm_powered"] is False
    assert "Answer:" in res["answer"]


def test_valid_mocked_llm_response(sample_inr_df):
    """Req 2: Valid mocked LLM API call formatting."""
    assistant = GroundedFinancialAssistant(provider="gemini", api_key="mock_key")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "**Answer:** Mocked summary for INR spending."
    mock_client.models.generate_content.return_value = mock_resp
    assistant.client = mock_client

    res = assistant.process_query("Where am I spending the most money?", sample_inr_df)
    assert res["is_llm_powered"] is True
    assert "Mocked summary" in res["answer"]


def test_provider_api_exception_fallback(sample_inr_df):
    """Req 3 & 4: API exception or timeout gracefully falls back to offline template."""
    assistant = GroundedFinancialAssistant(provider="gemini", api_key="mock_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Timeout / Network error")
    assistant.client = mock_client

    res = assistant.process_query("Where am I spending the most money?", sample_inr_df)
    assert "Answer:" in res["answer"]
    assert "House Rent" in res["answer"] or "Food & Dining" in res["answer"] or "Housing" in res["answer"]


def test_inr_formatting_no_hardcoded_dollar(sample_inr_df):
    """Req 10: INR formatting uses INR symbol without hardcoded dollar signs."""
    detector = SpendingAnomalyDetector()
    anom_df = detector.detect_anomalies(sample_inr_df)
    
    # Check that anomaly reasons do NOT contain hardcoded $ sign for INR transactions
    for reason in anom_df["anomaly_reason"]:
        if "High transaction amount" in reason:
            assert "$" not in reason
            assert "INR" in reason or "₹" in reason

    assert format_currency_amount(18000.0, "INR") == "INR 18,000.00"


def test_duplicate_anomaly_prevention_and_legitimate_distinct(sample_inr_df):
    """Req 9 & 11: Anomaly detector keeps distinct transactions separate and prevents duplicate row indexing."""
    res = ChatbotFinancialTools.detect_unusual_spending(sample_inr_df)
    assert res["status"] == "success"
    unusual = res["unusual_transactions"]
    
    # Verify that distinct transaction_ids are preserved
    tx_ids = [tx["transaction_id"] for tx in unusual]
    assert len(tx_ids) == len(set(tx_ids))  # Zero duplicate IDs


def test_followup_question_routing(sample_inr_df):
    """Req 7: Follow-up question context routing."""
    assistant = GroundedFinancialAssistant(api_key=None)
    history = [
        {
            "intent": "category_period_spending",
            "tool_result": {"status": "success", "category": "Food & Dining", "highest_spending_category": "Food & Dining"}
        }
    ]
    route = assistant.route_intent("Show those transactions", history=history)
    assert route["intent"] == "transaction_details"
    assert route["args"]["category"] == "Food & Dining"


def test_invalid_tool_args_graceful_handling(sample_inr_df):
    """Req 5 & 6: Invalid tool arguments are handled safely."""
    res = ChatbotFinancialTools.explain_transaction_classification("")
    assert res["status"] == "error"
    assert "valid transaction description" in res["message"]
