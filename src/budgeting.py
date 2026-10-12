import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class BudgetManager:
    """
    Deterministic budgeting engine.
    Calculates spending vs budgets per category and month, remaining balances,
    and over-budget / near-limit status flags.
    """

    def __init__(self, initial_budgets: Optional[Dict[str, Dict[str, float]]] = None):
        # Format: { "YYYY-MM": { "Category": budget_amount } }
        self.budgets: Dict[str, Dict[str, float]] = initial_budgets or {}

    def set_budget(self, month: str, category: str, amount: float) -> Dict[str, Any]:
        """
        Create or update a budget for a category and month.
        Validates month format (YYYY-MM) and amount (must be non-negative).
        """
        if not isinstance(month, str) or not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", month.strip()):
            raise ValueError(f"Invalid month format: '{month}'. Expected 'YYYY-MM' (e.g., '2026-08').")

        if not isinstance(category, str) or not category.strip():
            raise ValueError("Category must be a non-empty string.")

        try:
            val = float(amount)
        except (ValueError, TypeError):
            raise ValueError(f"Budget amount must be a valid number, got: {amount}")

        if val < 0:
            raise ValueError("Budget amount cannot be negative.")

        clean_month = month.strip()
        clean_cat = category.strip()

        if clean_month not in self.budgets:
            self.budgets[clean_month] = {}

        self.budgets[clean_month][clean_cat] = val
        return {
            "month": clean_month,
            "category": clean_cat,
            "amount": val,
            "status": "success"
        }

    def get_budget(self, month: str, category: str) -> float:
        """Retrieve budget for a specific month and category (returns 0.0 if not set)."""
        return self.budgets.get(month, {}).get(category, 0.0)

    def get_monthly_budgets(self, month: str) -> Dict[str, float]:
        """Retrieve all budgets set for a given month."""
        return self.budgets.get(month, {}).copy()

    @staticmethod
    def _extract_expenses(df: pd.DataFrame, month: Optional[str] = None) -> pd.DataFrame:
        """
        Filters and normalizes expenses for a given month.
        Documented expense rules:
        - If 'date' exists, filters to matching 'YYYY-MM'. Missing dates are excluded from month filter.
        - Excludes 'Income' category.
        - If 'transaction_type' exists:
          * 'expense' contributes to spending.
          * 'refund' offsets spending (reduces expense).
        - If amount is negative, abs(amount) is treated as expense.
        - If amount is positive and type is not 'refund' and category != 'Income', it is treated as expense.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["date", "category", "normalized_expense"])

        work_df = df.copy()

        # Handle category column
        cat_col = "category" if "category" in work_df.columns else "predicted_category"
        if cat_col not in work_df.columns:
            work_df[cat_col] = "Uncategorized"

        # Exclude Income
        work_df = work_df[work_df[cat_col].str.lower() != "income"].copy()

        # Handle date filter
        if "date" in work_df.columns and month is not None:
            work_df["date_str"] = pd.to_datetime(work_df["date"], errors="coerce").dt.strftime("%Y-%m")
            work_df = work_df[work_df["date_str"] == month].copy()

        # Calculate normalized expense amount
        if "amount" not in work_df.columns:
            work_df["normalized_expense"] = 0.0
            return work_df

        work_df["amount"] = pd.to_numeric(work_df["amount"], errors="coerce").fillna(0.0)

        # Check transaction_type
        if "transaction_type" in work_df.columns:
            types = work_df["transaction_type"].str.lower().fillna("expense")
            expenses = []
            for amt, t in zip(work_df["amount"], types):
                if t == "refund":
                    expenses.append(-abs(amt))
                elif amt < 0:
                    expenses.append(abs(amt))
                else:
                    expenses.append(amt)
            work_df["normalized_expense"] = expenses
        else:
            # When amount is negative (typical bank statements), abs is expense
            # If positive, also treated as expense unless marked
            work_df["normalized_expense"] = work_df["amount"].apply(lambda a: abs(a))

        work_df["clean_category"] = work_df[cat_col].fillna("Uncategorized")
        return work_df

    def calculate_actual_spending(self, df: pd.DataFrame, month: str) -> Dict[str, float]:
        """Calculates actual spending per category for a given month."""
        exp_df = self._extract_expenses(df, month=month)
        if len(exp_df) == 0:
            return {}

        grouped = exp_df.groupby("clean_category")["normalized_expense"].sum()
        # Spending cannot be negative overall; clamp at 0.0
        return {cat: max(0.0, float(round(amt, 2))) for cat, amt in grouped.items()}

    def generate_budget_report(self, df: pd.DataFrame, month: str, near_limit_ratio: float = 0.80) -> pd.DataFrame:
        """
        Generates a consolidated budget performance report.
        Flags:
        - Within Budget (< near_limit_ratio, e.g. < 80%)
        - Near Limit (>= near_limit_ratio and <= 100%)
        - Over Budget (> 100%)
        - Unbudgeted (spent > 0 with no budget defined)
        """
        budgets_for_month = self.get_monthly_budgets(month)
        actual_spending = self.calculate_actual_spending(df, month)

        # Combine all categories from either budgets or actual spending
        all_categories = sorted(list(set(budgets_for_month.keys()).union(set(actual_spending.keys()))))

        rows = []
        for cat in all_categories:
            budget = budgets_for_month.get(cat, 0.0)
            actual = actual_spending.get(cat, 0.0)
            remaining = budget - actual
            pct = (actual / budget * 100.0) if budget > 0 else (np.inf if actual > 0 else 0.0)

            if budget == 0.0 and actual > 0.0:
                status = "Unbudgeted"
            elif actual > budget:
                status = "Over Budget"
            elif budget > 0 and (actual / budget) >= near_limit_ratio:
                status = "Near Limit"
            else:
                status = "Within Budget"

            rows.append({
                "Month": month,
                "Category": cat,
                "Budget": round(budget, 2),
                "Actual_Spent": round(actual, 2),
                "Remaining": round(remaining, 2),
                "Percentage_Used": round(pct, 2) if not np.isinf(pct) else 999.9,
                "Status": status
            })

        if not rows:
            return pd.DataFrame(columns=["Month", "Category", "Budget", "Actual_Spent", "Remaining", "Percentage_Used", "Status"])

        return pd.DataFrame(rows)
