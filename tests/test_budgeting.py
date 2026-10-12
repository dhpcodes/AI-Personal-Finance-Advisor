import pytest
import pandas as pd
from src.budgeting import BudgetManager

def test_set_and_get_budget():
    bm = BudgetManager()
    res = bm.set_budget("2026-08", "Food & Dining", 15000.0)
    assert res["status"] == "success"
    assert bm.get_budget("2026-08", "Food & Dining") == 15000.0
    assert bm.get_budget("2026-08", "Entertainment & Recreation") == 0.0

def test_invalid_budget_inputs():
    bm = BudgetManager()
    with pytest.raises(ValueError, match="Invalid month format"):
        bm.set_budget("2026-13", "Food", 1000.0)

    with pytest.raises(ValueError, match="Category must be a non-empty string"):
        bm.set_budget("2026-08", "", 1000.0)

    with pytest.raises(ValueError, match="cannot be negative"):
        bm.set_budget("2026-08", "Food", -500.0)

def test_actual_spending_and_refunds():
    # Hand-crafted transactions with known expected results
    txns = [
        {"date": "2026-08-01", "category": "Food & Dining", "amount": -1200.0, "transaction_type": "expense"},
        {"date": "2026-08-05", "category": "Food & Dining", "amount": -800.0, "transaction_type": "expense"},
        {"date": "2026-08-10", "category": "Food & Dining", "amount": 200.0, "transaction_type": "refund"}, # refund of 200
        {"date": "2026-08-12", "category": "Shopping & Retail", "amount": -4500.0, "transaction_type": "expense"},
        {"date": "2026-08-01", "category": "Income", "amount": 50000.0, "transaction_type": "income"}, # Income excluded
        {"date": "2026-09-01", "category": "Food & Dining", "amount": -1000.0, "transaction_type": "expense"}, # Different month
    ]
    df = pd.DataFrame(txns)
    bm = BudgetManager()

    spending_aug = bm.calculate_actual_spending(df, "2026-08")
    # Food: 1200 + 800 - 200 = 1800.0
    assert spending_aug["Food & Dining"] == 1800.0
    # Shopping: 4500.0
    assert spending_aug["Shopping & Retail"] == 4500.0
    # Income excluded
    assert "Income" not in spending_aug

def test_budget_report_status_flags():
    txns = [
        {"date": "2026-08-01", "category": "Food & Dining", "amount": -9500.0},        # 95% of 10000 -> Near Limit
        {"date": "2026-08-05", "category": "Transportation", "amount": -3000.0},       # 30% of 10000 -> Within Budget
        {"date": "2026-08-10", "category": "Shopping & Retail", "amount": -6500.0},    # 130% of 5000 -> Over Budget
        {"date": "2026-08-15", "category": "Entertainment & Recreation", "amount": -1200.0}, # No budget -> Unbudgeted
    ]
    df = pd.DataFrame(txns)
    bm = BudgetManager()
    bm.set_budget("2026-08", "Food & Dining", 10000.0)
    bm.set_budget("2026-08", "Transportation", 10000.0)
    bm.set_budget("2026-08", "Shopping & Retail", 5000.0)

    report = bm.generate_budget_report(df, "2026-08", near_limit_ratio=0.80)
    status_map = dict(zip(report["Category"], report["Status"]))

    assert status_map["Food & Dining"] == "Near Limit"
    assert status_map["Transportation"] == "Within Budget"
    assert status_map["Shopping & Retail"] == "Over Budget"
    assert status_map["Entertainment & Recreation"] == "Unbudgeted"

def test_empty_dataframe():
    bm = BudgetManager()
    bm.set_budget("2026-08", "Food & Dining", 5000.0)
    report = bm.generate_budget_report(pd.DataFrame(), "2026-08")
    assert len(report) == 1
    assert report.iloc[0]["Actual_Spent"] == 0.0
    assert report.iloc[0]["Status"] == "Within Budget"
