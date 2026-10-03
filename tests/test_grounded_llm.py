import pytest
import pandas as pd
from src.grounded_llm import GroundedFinancialAssistant


@pytest.fixture
def sample_df():
    data = [
        {"date": "2026-08-01", "transaction_description": "Salary Direct Deposit", "amount": 5000.0, "category": "Income"},
        {"date": "2026-08-05", "transaction_description": "Starbucks Coffee #1029", "amount": -15.50, "category": "Food & Dining"},
        {"date": "2026-08-15", "transaction_description": "Netflix Subscription", "amount": -19.99, "category": "Entertainment & Recreation"},
        {"date": "2026-09-01", "transaction_description": "Salary Direct Deposit", "amount": 5000.0, "category": "Income"},
        {"date": "2026-09-03", "transaction_description": "Starbucks Coffee #1029", "amount": -14.20, "category": "Food & Dining"},
        {"date": "2026-09-15", "transaction_description": "Netflix Subscription", "amount": -19.99, "category": "Entertainment & Recreation"},
        {"date": "2026-09-18", "transaction_description": "Luxury Watch Purchase", "amount": -2500.00, "category": "Shopping & Retail"}
    ]
    return pd.DataFrame(data)


def test_intent_routing():
    assistant = GroundedFinancialAssistant(api_key=None)
    
    r1 = assistant.route_intent("Where am I spending the most money?")
    assert r1["intent"] == "spending_breakdown"

    r2 = assistant.route_intent("How much did I spend on food this month?")
    assert r2["intent"] == "category_period_spending"
    assert r2["args"]["category"] == "Food & Dining"

    r3 = assistant.route_intent("What are my recurring expenses?")
    assert r3["intent"] == "recurring_expenses"

    r4 = assistant.route_intent("Why was 'Netflix Subscription' classified as entertainment?")
    assert r4["intent"] == "explain_classification"

    r5 = assistant.route_intent("What could my expenses look like next month?")
    assert r5["intent"] == "forecast_expenses"


def test_process_query_offline(sample_df):
    assistant = GroundedFinancialAssistant(api_key=None)
    
    res = assistant.process_query("Where am I spending the most money?", sample_df)
    assert res["is_llm_powered"] is False
    assert "Answer:" in res["answer"]
    assert "Shopping & Retail" in res["answer"] or "Food & Dining" in res["answer"]
    assert "Details:" in res["answer"]
    assert "Method:" in res["answer"]


def test_process_query_forecast(sample_df):
    assistant = GroundedFinancialAssistant(api_key=None)
    
    res = assistant.process_query("Predict next month expenses", sample_df)
    assert res["intent"] == "forecast_expenses"
    assert "Forecast" in res["answer"] or "Estimated" in res["answer"]


def test_recurring_expense_followup_sequence(sample_df):
    """Regression test for the exact two-question sequence:
    1. 'What are my recurring expenses?'
    2. 'Which recurring expense has the highest average amount?'
    """
    assistant = GroundedFinancialAssistant(api_key=None)
    history = []

    # Question 1
    res1 = assistant.process_query("What are my recurring expenses?", sample_df, history=history)
    assert res1["intent"] == "recurring_expenses"
    assert res1["tool_result"]["status"] == "success"
    assert res1["tool_result"]["recurring_expenses_count"] > 0

    # Simulate history tracking as in chat application
    history.append({"role": "user", "content": "What are my recurring expenses?"})
    history.append({
        "role": "assistant",
        "content": res1["answer"],
        "details": res1["tool_result"],
        "intent": res1["intent"],
        "tool_result": res1["tool_result"],
    })

    # Question 2
    res2 = assistant.process_query("Which recurring expense has the highest average amount?", sample_df, history=history)
    assert res2["intent"] == "highest_recurring_expense"
    assert res2["tool_result"]["status"] == "success"

    # Verify highest average expense is dynamically calculated from tool results
    rec_list = res1["tool_result"]["recurring_expenses"]
    expected_highest = max(rec_list, key=lambda x: x["average_amount"])
    assert res2["tool_result"]["merchant"] == expected_highest["merchant"]
    assert res2["tool_result"]["average_amount"] == expected_highest["average_amount"]

    assert expected_highest["merchant"] in res2["answer"]
    assert "highest average amount" in res2["answer"]


def test_highest_recurring_expense_unavailable_without_history(sample_df):
    """Req 5: Asking follow-up question without prior history returns unavailable message instead of repeating full report."""
    assistant = GroundedFinancialAssistant(api_key=None)
    res = assistant.process_query("Which recurring expense has the highest average amount?", sample_df, history=[])
    assert res["intent"] == "highest_recurring_expense"
    assert res["tool_result"]["status"] == "unavailable"
    assert "Previous recurring expense results are unavailable" in res["answer"]

