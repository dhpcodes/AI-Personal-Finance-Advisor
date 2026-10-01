import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils import logger

class FinanceAnalytics:
    """Calculates personal finance summaries, category breakdowns, monthly trends, and recurring patterns."""

    @staticmethod
    def calculate_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute top-level summary metrics (income, expenses, net cash flow, currency groups)."""
        summary = {
            "total_transactions": len(df),
            "currencies_found": list(df["currency"].unique()) if "currency" in df.columns else ["USD"],
            "by_currency": {}
        }
        
        # Determine amount and income vs expense
        if "amount" not in df.columns:
            summary["has_amount_data"] = False
            summary["message"] = "Amount data not present in file. Summary restricted to transaction counts."
            return summary

        summary["has_amount_data"] = True
        df = df.copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        
        # Currency breakdown
        currency_col = "currency" if "currency" in df.columns else None
        
        if currency_col:
            currencies = df[currency_col].unique()
        else:
            currencies = ["USD"]
            df["currency"] = "USD"

        for curr in currencies:
            curr_df = df[df["currency"] == curr]
            
            # Determine income vs expense
            if "category" in curr_df.columns:
                income_mask = curr_df["category"].astype(str).str.lower() == "income"
                income = curr_df[income_mask]["amount"].abs().sum()
                expense = curr_df[~income_mask]["amount"].abs().sum()
            else:
                income = curr_df[curr_df["amount"] > 0]["amount"].sum()
                expense = curr_df[curr_df["amount"] < 0]["amount"].abs().sum()
                
            net_flow = income - expense
            
            summary["by_currency"][curr] = {
                "total_income": round(float(income), 2),
                "total_expense": round(float(expense), 2),
                "net_cash_flow": round(float(net_flow), 2),
                "transaction_count": len(curr_df)
            }
            
        return summary

    @staticmethod
    def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate category-wise spending and percentage breakdown."""
        if "predicted_category" in df.columns:
            cat_col = "predicted_category"
        elif "category" in df.columns:
            cat_col = "category"
        else:
            return pd.DataFrame()

        df_exp = df[df[cat_col].astype(str).str.lower() != "income"].copy() if "amount" in df.columns else df.copy()
        
        if "amount" in df_exp.columns:
            df_exp["amount"] = pd.to_numeric(df_exp["amount"], errors="coerce").abs()
            cat_summary = df_exp.groupby(cat_col).agg(
                Total_Amount=("amount", "sum"),
                Transaction_Count=("amount", "count")
            ).reset_index()
            total_sum = cat_summary["Total_Amount"].sum()
            cat_summary["Percentage (%)"] = (
                (cat_summary["Total_Amount"] / total_sum * 100).round(2) if total_sum > 0 else 0.0
            )
            cat_summary.sort_values(by="Total_Amount", ascending=False, inplace=True)
        else:
            cat_summary = df_exp.groupby(cat_col).size().reset_index(name="Transaction_Count")
            total_cnt = cat_summary["Transaction_Count"].sum()
            cat_summary["Percentage (%)"] = ((cat_summary["Transaction_Count"] / total_cnt) * 100).round(2)
            cat_summary.sort_values(by="Transaction_Count", ascending=False, inplace=True)

        return cat_summary

    @staticmethod
    def detect_recurring_transactions(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
        """Identify recurring merchants or transaction descriptions."""
        desc_col = "transaction_description" if "transaction_description" in df.columns else "description"
        if desc_col not in df.columns:
            return pd.DataFrame()

        counts = df.groupby(desc_col).size().reset_index(name="Frequency")
        recurring = counts[counts["Frequency"] >= min_count].sort_values(by="Frequency", ascending=False)
        
        if "amount" in df.columns and len(recurring) > 0:
            avg_amounts = df.groupby(desc_col)["amount"].mean().abs().round(2).reset_index(name="Avg_Amount")
            recurring = recurring.merge(avg_amounts, on=desc_col)
            
        return recurring

    @staticmethod
    def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Compute monthly income and expense trends if date column exists."""
        if "date" not in df.columns or "amount" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
        if len(df) == 0:
            return pd.DataFrame()

        df["YearMonth"] = df["date"].dt.to_period("M").astype(str)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        
        if "category" in df.columns:
            df["is_income"] = df["category"].astype(str).str.lower() == "income"
        else:
            df["is_income"] = df["amount"] > 0

        trends = (
    df.assign(amount_abs=df["amount"].abs())
      .groupby(["YearMonth", "is_income"])["amount_abs"]
      .sum()
      .unstack(fill_value=0.0)
      .reset_index()
)
        
        # Rename columns safely
        trends.rename(columns={True: "Income", False: "Expense"}, inplace=True)
        if "Income" not in trends.columns:
            trends["Income"] = 0.0
        if "Expense" not in trends.columns:
            trends["Expense"] = 0.0
            
        trends["Net_Cash_Flow"] = (trends["Income"] - trends["Expense"]).round(2)
        return trends
