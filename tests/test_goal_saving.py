import pytest
import pandas as pd
from src.goal_saving import GoalSavingPlanner

def test_goal_saving_arithmetic():
    # Target 120,000, Current savings 20,000, 10 months remaining
    # Remaining required = 100,000. Required monthly = 10,000
    res = GoalSavingPlanner.evaluate_goal(
        target_amount=120000.0,
        current_savings=20000.0,
        months_remaining=10
    )
    assert res["status"] == "in_progress"
    assert res["remaining_required"] == 100000.0
    assert res["required_monthly_saving"] == 10000.0

def test_goal_already_achieved():
    res = GoalSavingPlanner.evaluate_goal(
        target_amount=50000.0,
        current_savings=60000.0,
        months_remaining=5
    )
    assert res["status"] == "achieved"
    assert res["remaining_required"] == 0.0
    assert res["required_monthly_saving"] == 0.0

def test_goal_invalid_inputs():
    with pytest.raises(ValueError, match="strictly positive"):
        GoalSavingPlanner.evaluate_goal(target_amount=-100.0, current_savings=0, months_remaining=5)

    with pytest.raises(ValueError, match="cannot be negative"):
        GoalSavingPlanner.evaluate_goal(target_amount=1000.0, current_savings=-50, months_remaining=5)

    with pytest.raises(ValueError, match="greater than zero"):
        GoalSavingPlanner.evaluate_goal(target_amount=1000.0, current_savings=0, months_remaining=0)

def test_goal_feasibility_with_history():
    # 2 months of history:
    # Month 1: Income 80,000, Food 15,000, Utilities 5,000 -> Expense 20,000. Surplus = 60,000
    # Month 2: Income 80,000, Food 15,000, Shopping 15,000 -> Expense 30,000. Surplus = 50,000
    # Average surplus = 55,000
    txns = [
        {"date": "2026-07-01", "category": "Income", "amount": 80000.0},
        {"date": "2026-07-05", "category": "Food & Dining", "amount": -15000.0},
        {"date": "2026-07-10", "category": "Utilities & Services", "amount": -5000.0},
        {"date": "2026-08-01", "category": "Income", "amount": 80000.0},
        {"date": "2026-08-05", "category": "Food & Dining", "amount": -15000.0},
        {"date": "2026-08-12", "category": "Shopping & Retail", "amount": -15000.0},
    ]
    df = pd.DataFrame(txns)

    # Goal A: Target 100,000 in 5 months -> 20,000/mo. Surplus 55,000 >= 20,000 -> Feasible!
    res_a = GoalSavingPlanner.evaluate_goal(
        target_amount=100000.0,
        current_savings=0.0,
        months_remaining=5,
        df=df
    )
    assert res_a["is_feasible"] is True
    assert res_a["surplus_margin"] == 35000.0

    # Goal B: Target 600,000 in 6 months -> 100,000/mo. Surplus 55,000 < 100,000 -> Shortfall!
    res_b = GoalSavingPlanner.evaluate_goal(
        target_amount=600000.0,
        current_savings=0.0,
        months_remaining=6,
        df=df
    )
    assert res_b["is_feasible"] is False
    assert res_b["estimated_shortfall"] == 45000.0
    assert len(res_b["recommendations"]) > 0
