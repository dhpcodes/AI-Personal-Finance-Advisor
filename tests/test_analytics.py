import pytest
import pandas as pd
from src.analytics import FinanceAnalytics
from src.anomaly_detection import SpendingAnomalyDetector

def test_calculate_summary():
    df = pd.DataFrame({
        "transaction_description": ["Salary", "Coffee", "Electricity Bill"],
        "category": ["Income", "Food & Dining", "Utilities & Services"],
        "amount": [5000.0, 5.50, 120.0],
        "currency": ["USD", "USD", "USD"]
    })
    summary = FinanceAnalytics.calculate_summary(df)
    assert summary["has_amount_data"] is True
    usd_summary = summary["by_currency"]["USD"]
    assert usd_summary["total_income"] == 5000.0
    assert usd_summary["total_expense"] == 125.50
    assert usd_summary["net_cash_flow"] == 4874.50

def test_category_breakdown():
    df = pd.DataFrame({
        "category": ["Food & Dining", "Food & Dining", "Shopping & Retail"],
        "amount": [20.0, 30.0, 50.0]
    })
    breakdown = FinanceAnalytics.get_category_breakdown(df)
    assert len(breakdown) == 2
    assert "Percentage (%)" in breakdown.columns

def test_anomaly_detection():
    detector = SpendingAnomalyDetector()
    df = pd.DataFrame({
        "transaction_description": [f"Normal Txn {i}" for i in range(20)] + ["Massive Outlier Purchase"],
        "amount": [20.0 + i for i in range(20)] + [15000.0]
    })
    res_df = detector.detect_anomalies(df)
    assert "is_unusual" in res_df.columns
    assert res_df.iloc[-1]["is_unusual"] is True or res_df["is_unusual"].sum() > 0
