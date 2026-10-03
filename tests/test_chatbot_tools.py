
import pytest
import pandas as pd
import numpy as np
from src.chatbot_tools import ChatbotFinancialTools


@pytest.fixture
def sample_financial_df():
    """Create a multi-month sample transaction DataFrame for testing."""
    data = [
        {"date": "2026-08-01", "transaction_description": "Salary Direct Deposit", "amount": 5000.0, "category": "Income", "currency": "USD"},
        {"date": "2026-08-05", "transaction_description": "Starbucks Coffee #1029", "amount": -15.50, "category": "Food & Dining", "currency": "USD"},
        {"date": "2026-08-10", "transaction_description": "Uber Trip Ride", "amount": -45.00, "category": "Transportation", "currency": "USD"},
        {"date": "2026-08-15", "transaction_description": "Netflix Subscription", "amount": -19.99, "category": "Entertainment & Recreation", "currency": "USD"},
        {"date": "2026-08-20", "transaction_description": "Whole Foods Market", "amount": -185.20, "category": "Food & Dining", "currency": "USD"},
        {"date": "2026-09-01", "transaction_description": "Salary Direct Deposit", "amount": 5000.0, "category": "Income", "currency": "USD"},
        {"date": "2026-09-03", "transaction_description": "Starbucks Coffee #1029", "amount": -14.20, "category": "Food & Dining", "currency": "USD"},
        {"date": "2026-09-15", "transaction_description": "Netflix Subscription", "amount": -19.99, "category": "Entertainment & Recreation", "currency": "USD"},
        {"date": "2026-09-18", "transaction_description": "Luxury Watch Purchase", "amount": -2500.00, "category": "Shopping & Retail", "currency": "USD"},
        {"date": "2026-09-22", "transaction_description": "Chipotle Mexican Grill", "amount": -22.50, "category": "Food & Dining", "currency": "USD"}
    ]
    return pd.DataFrame(data)


def test_get_spending_by_category(sample_financial_df):
    res = ChatbotFinancialTools.get_spending_by_category(sample_financial_df)
    assert res["status"] == "success"
    assert res["total_expense"] > 0
    assert res["highest_spending_category"] == "Shopping & Retail"
    assert len(res["category_breakdown"]) > 0


def test_get_category_spending_for_period(sample_financial_df):
    res = ChatbotFinancialTools.get_category_spending_for_period(
        sample_financial_df, category="Food & Dining"
    )
    assert res["status"] == "success"
    assert res["current_spending"] > 0
    assert res["transaction_count"] > 0
    assert res["has_previous_period"] is True


def test_get_recurring_expenses(sample_financial_df):
    res = ChatbotFinancialTools.get_recurring_expenses(
        sample_financial_df, min_count=2
    )
    assert res["status"] == "success"
    assert res["recurring_expenses_count"] >= 2
    merchants = [item["merchant"] for item in res["recurring_expenses"]]
    assert (
        "Netflix Subscription" in merchants
        or "Starbucks Coffee #1029" in merchants
    )


def test_explain_transaction_classification():
    res = ChatbotFinancialTools.explain_transaction_classification(
        "Netflix Subscription Renewal #4892"
    )
    assert res["status"] == "success"
    assert "predicted_category" in res
    assert "confidence_score" in res


def test_forecast_monthly_expenses(sample_financial_df):
    res = ChatbotFinancialTools.forecast_monthly_expenses(
        sample_financial_df, min_months=2
    )
    assert res["status"] == "success"
    assert res["predicted_amount"] > 0
    assert res["upper_bound"] >= res["predicted_amount"]


def test_forecast_insufficient_data():
    single_month_df = pd.DataFrame([
        {
            "date": "2026-09-01",
            "transaction_description": "Coffee",
            "amount": -5.0,
            "category": "Food & Dining",
        }
    ])
    res = ChatbotFinancialTools.forecast_monthly_expenses(
        single_month_df, min_months=2
    )
    assert res["status"] == "insufficient_data"
    assert "Not enough historical data" in res["message"]


def test_get_monthly_income_expense(sample_financial_df):
    res = ChatbotFinancialTools.get_monthly_income_expense(
        sample_financial_df
    )
    assert res["status"] == "success"
    assert len(res["monthly_cash_flow"]) == 2


def test_detect_unusual_spending(sample_financial_df):
    res = ChatbotFinancialTools.detect_unusual_spending(sample_financial_df)
    assert res["status"] == "success"
    assert res["unusual_transactions_count"] > 0
    descriptions = [
        item["description"] for item in res["unusual_transactions"]
    ]
    assert "Luxury Watch Purchase" in descriptions


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    res = ChatbotFinancialTools.get_spending_by_category(empty_df)
    assert res["status"] == "error"
    assert "No transaction data" in res["message"]


def test_existing_category_takes_priority_over_model_prediction():
    """Prefer the category supplied in the CSV over the model prediction."""
    df = pd.DataFrame([
        {
            "date": "2026-09-10",
            "transaction_description": "ZOMATO ORDER PAYMENT",
            "amount": -350.0,
            "category": "Food & Dining",
            "predicted_category": "Shopping & Retail",
            "currency": "INR",
        }
    ])

    res = ChatbotFinancialTools.get_category_spending_for_period(
        df, category="Food & Dining"
    )

    assert res["status"] == "success"
    assert res["current_spending"] == pytest.approx(350.0)
    assert res["transaction_count"] == 1


def test_get_highest_recurring_expense(sample_financial_df):
    rec_res = ChatbotFinancialTools.get_recurring_expenses(sample_financial_df)
    assert rec_res["status"] == "success"

    highest_res = ChatbotFinancialTools.get_highest_recurring_expense(previous_result=rec_res)
    assert highest_res["status"] == "success"
    assert highest_res["average_amount"] > 0
    assert highest_res["merchant"] in ["Netflix Subscription", "Starbucks Coffee #1029"]


def test_get_highest_recurring_expense_unavailable():
    res = ChatbotFinancialTools.get_highest_recurring_expense(previous_result=None)
    assert res["status"] == "unavailable"
    assert "unavailable" in res["message"]


