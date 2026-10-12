from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import pandas as pd
import numpy as np

class GoalSavingPlanner:
    """
    Deterministic goal-based saving planning module.
    Uses pure arithmetic and historical transaction cash-flow analytics.
    Does NOT depend on an LLM or make financial guarantees.
    """

    DISCRETIONARY_CATEGORIES = [
        "Food & Dining",
        "Entertainment & Recreation",
        "Shopping & Retail"
    ]

    @staticmethod
    def calculate_historical_surplus(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates average monthly income, spending, net cash flow surplus,
        and discretionary category breakdown from historical transactions.
        """
        if df is None or len(df) == 0 or "amount" not in df.columns:
            return {
                "status": "insufficient_data",
                "message": "No transaction history available to compute historical cash flow surplus.",
                "has_history": False,
                "average_monthly_income": 0.0,
                "average_monthly_expense": 0.0,
                "average_monthly_surplus": 0.0,
                "discretionary_spending": {}
            }

        work_df = df.copy()
        work_df["amount"] = pd.to_numeric(work_df["amount"], errors="coerce").fillna(0.0)

        # Require dates for monthly aggregation
        if "date" not in work_df.columns:
            return {
                "status": "insufficient_data",
                "message": "Missing 'date' column in transaction history.",
                "has_history": False,
                "average_monthly_income": 0.0,
                "average_monthly_expense": 0.0,
                "average_monthly_surplus": 0.0,
                "discretionary_spending": {}
            }

        work_df["YearMonth"] = pd.to_datetime(work_df["date"], errors="coerce").dt.strftime("%Y-%m")
        work_df = work_df.dropna(subset=["YearMonth"])
        unique_months = work_df["YearMonth"].nunique()

        if unique_months == 0:
            return {
                "status": "insufficient_data",
                "message": "No valid dates found in transaction history.",
                "has_history": False,
                "average_monthly_income": 0.0,
                "average_monthly_expense": 0.0,
                "average_monthly_surplus": 0.0,
                "discretionary_spending": {}
            }

        cat_col = "category" if "category" in work_df.columns else "predicted_category"
        if cat_col not in work_df.columns:
            work_df[cat_col] = "Uncategorized"

        # Separate Income and Expense
        is_income = (work_df[cat_col].str.lower() == "income")
        if "transaction_type" in work_df.columns:
            is_income = is_income | (work_df["transaction_type"].str.lower() == "income")

        total_income = work_df[is_income]["amount"].apply(abs).sum()
        total_expense = work_df[~is_income]["amount"].apply(abs).sum()

        avg_income = float(round(total_income / unique_months, 2))
        avg_expense = float(round(total_expense / unique_months, 2))
        avg_surplus = float(round(avg_income - avg_expense, 2))

        # Discretionary spending breakdown
        disc_df = work_df[work_df[cat_col].isin(GoalSavingPlanner.DISCRETIONARY_CATEGORIES)]
        disc_grouped = disc_df.groupby(cat_col)["amount"].apply(lambda s: s.apply(abs).sum() / unique_months)
        disc_dict = {cat: float(round(val, 2)) for cat, val in disc_grouped.items()}

        return {
            "status": "success",
            "has_history": True,
            "months_analyzed": unique_months,
            "average_monthly_income": avg_income,
            "average_monthly_expense": avg_expense,
            "average_monthly_surplus": avg_surplus,
            "discretionary_spending": disc_dict
        }

    @staticmethod
    def evaluate_goal(
        target_amount: float,
        current_savings: float,
        months_remaining: Union[int, float],
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a savings goal against mathematical targets and historical cash flow.
        """
        # 1. Validation
        try:
            target = float(target_amount)
            savings = float(current_savings)
            months = float(months_remaining)
        except (ValueError, TypeError) as ex:
            raise ValueError(f"Target amount, current savings, and months must be numeric: {ex}")

        if target <= 0:
            raise ValueError("Target goal amount must be strictly positive.")
        if savings < 0:
            raise ValueError("Current savings cannot be negative.")
        if months <= 0:
            raise ValueError("Months remaining must be greater than zero.")

        # 2. Arithmetic calculations
        remaining_required = max(0.0, target - savings)
        if remaining_required == 0.0:
            return {
                "status": "achieved",
                "target_amount": target,
                "current_savings": savings,
                "remaining_required": 0.0,
                "months_remaining": months,
                "required_monthly_saving": 0.0,
                "is_feasible": True,
                "message": "Savings goal is already fully achieved! No additional monthly saving required."
            }

        required_monthly = round(remaining_required / months, 2)

        result: Dict[str, Any] = {
            "status": "in_progress",
            "target_amount": target,
            "current_savings": savings,
            "remaining_required": remaining_required,
            "months_remaining": months,
            "required_monthly_saving": required_monthly,
            "is_feasible": None,
            "estimated_shortfall": 0.0,
            "recommendations": []
        }

        # 3. Assess feasibility if transaction history is available
        if df is not None:
            history = GoalSavingPlanner.calculate_historical_surplus(df)
            result["history_analysis"] = history

            if history.get("has_history"):
                avg_surplus = history["average_monthly_surplus"]
                if avg_surplus >= required_monthly:
                    result["is_feasible"] = True
                    result["surplus_margin"] = round(avg_surplus - required_monthly, 2)
                    result["summary"] = (
                        f"Feasible: Your historical monthly surplus of ₹{avg_surplus:,.2f} "
                        f"exceeds the required monthly saving of ₹{required_monthly:,.2f} by ₹{result['surplus_margin']:,.2f}/month."
                    )
                else:
                    result["is_feasible"] = False
                    shortfall = round(required_monthly - avg_surplus, 2)
                    result["estimated_shortfall"] = shortfall
                    result["summary"] = (
                        f"Shortfall Identified: Based on your historical surplus of ₹{avg_surplus:,.2f}, "
                        f"you have an estimated monthly shortfall of ₹{shortfall:,.2f} to reach ₹{required_monthly:,.2f}/month."
                    )

                    # Suggest discretionary cuts
                    disc = history.get("discretionary_spending", {})
                    recs = []
                    for cat, amt in disc.items():
                        if amt > 0:
                            recs.append(f"Consider reducing {cat} spending (currently approx ₹{amt:,.2f}/month).")
                    result["recommendations"] = recs
            else:
                result["summary"] = (
                    f"To reach your goal of ₹{target:,.2f} in {months:.0f} months, "
                    f"you need to save ₹{required_monthly:,.2f} per month. "
                    "Insufficient transaction history to assess past monthly cash flow surplus."
                )
        else:
            result["summary"] = (
                f"To reach your goal of ₹{target:,.2f} in {months:.0f} months, "
                f"you need to save ₹{required_monthly:,.2f} per month."
            )

        return result
