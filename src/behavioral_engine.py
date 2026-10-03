import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.database import db_manager
from src.anomaly_detection import SpendingAnomalyDetector
from src.utils import logger

class BehavioralEngine:
    """Computes structured behavioral profiles for users from historical transactions."""

    def __init__(self, anomaly_detector: SpendingAnomalyDetector = None):
        self.anomaly_detector = anomaly_detector or SpendingAnomalyDetector()

    def compute_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Compute comprehensive UserBehaviorProfile for given user_id."""
        tx_df = db_manager.get_user_transactions(user_id)
        
        if len(tx_df) == 0:
            empty_profile = {
                "user_id": user_id,
                "calculated_at": datetime.now().isoformat(),
                "total_transactions": 0,
                "total_income": 0.0,
                "total_expense": 0.0,
                "net_cash_flow": 0.0,
                "average_transaction": 0.0,
                "median_transaction": 0.0,
                "max_transaction": 0.0,
                "spending_variability": 0.0,
                "category_distribution": {},
                "top_categories": [],
                "recurring_merchants": [],
                "monthly_spending": {},
                "spending_trend": "No transaction history",
                "category_concentration_pct": 0.0,
                "anomaly_frequency": 0
            }
            db_manager.save_behavior_profile(user_id, empty_profile)
            return empty_profile

        tx_df = tx_df.copy()
        tx_df["amount"] = pd.to_numeric(tx_df["amount"], errors="coerce").fillna(0.0)
        
        # Determine income vs expense
        category_col = "final_category" if "final_category" in tx_df.columns else "predicted_category"
        is_income_mask = (tx_df[category_col].astype(str).str.lower() == "income") | (tx_df["transaction_type"] == "income")
        
        income_df = tx_df[is_income_mask]
        expense_df = tx_df[~is_income_mask]
        
        total_income = round(float(income_df["amount"].abs().sum()), 2)
        total_expense = round(float(expense_df["amount"].abs().sum()), 2)
        net_cash_flow = round(total_income - total_expense, 2)
        
        expense_amounts = expense_df["amount"].abs()
        avg_tx = round(float(expense_amounts.mean()), 2) if len(expense_amounts) > 0 else 0.0
        med_tx = round(float(expense_amounts.median()), 2) if len(expense_amounts) > 0 else 0.0
        max_tx = round(float(expense_amounts.max()), 2) if len(expense_amounts) > 0 else 0.0
        std_tx = round(float(expense_amounts.std()), 2) if len(expense_amounts) > 1 else 0.0

        # Category distribution
        cat_dist = {}
        if len(expense_df) > 0:
            cat_sums = expense_df.groupby(category_col)["amount"].abs().sum()
            tot_exp_sum = cat_sums.sum()
            for cat, val in cat_sums.items():
                pct = round((val / tot_exp_sum * 100), 2) if tot_exp_sum > 0 else 0.0
                cat_dist[cat] = {"amount": round(float(val), 2), "percentage": pct}

        top_cats = sorted(cat_dist.keys(), key=lambda c: cat_dist[c]["amount"], reverse=True)[:3]
        
        # Category concentration (Top 2 category % sum)
        top2_sum = sum([cat_dist[c]["percentage"] for c in top_cats[:2]]) if len(top_cats) >= 2 else (
            cat_dist[top_cats[0]]["percentage"] if len(top_cats) == 1 else 0.0
        )

        # Recurring merchants (frequency >= 2)
        rec_merchants = []
        if "description" in expense_df.columns:
            counts = expense_df.groupby("description").size()
            recurring = counts[counts >= 2].reset_index(name="frequency")
            for _, r in recurring.iterrows():
                m_desc = r["description"]
                m_avg = expense_df[expense_df["description"] == m_desc]["amount"].abs().mean()
                rec_merchants.append({
                    "merchant": m_desc,
                    "frequency": int(r["frequency"]),
                    "avg_amount": round(float(m_avg), 2)
                })

        # Anomaly detection count
        anom_count = 0
        if len(expense_df) >= 5:
            anom_df = self.anomaly_detector.detect_anomalies(expense_df)
            anom_count = int(anom_df["is_unusual"].sum())

        # Monthly trend calculation
        monthly_spending = {}
        tx_df["date_parsed"] = pd.to_datetime(tx_df["date"], errors="coerce")
        valid_dates = tx_df.dropna(subset=["date_parsed"]).copy()
        
        spending_trend_desc = "Stable spending"
        if len(valid_dates) > 0:
            valid_dates["month"] = valid_dates["date_parsed"].dt.to_period("M").astype(str)
            month_exp = valid_dates[~is_income_mask].groupby("month")["amount"].abs().sum()
            monthly_spending = {m: round(float(v), 2) for m, v in month_exp.items()}
            
            if len(month_exp) >= 2:
                recent_months = list(month_exp.values())[-2:]
                prev, curr = recent_months[0], recent_months[1]
                if prev > 0:
                    pct_change = round(((curr - prev) / prev * 100), 1)
                    if pct_change > 10:
                        spending_trend_desc = f"Increased by {pct_change}% over previous month"
                    elif pct_change < -10:
                        spending_trend_desc = f"Decreased by {abs(pct_change)}% over previous month"
                    else:
                        spending_trend_desc = f"Relatively stable ({pct_change}% change)"

        profile = {
            "user_id": user_id,
            "calculated_at": datetime.now().isoformat(),
            "total_transactions": len(tx_df),
            "total_income": total_income,
            "total_expense": total_expense,
            "net_cash_flow": net_cash_flow,
            "average_transaction": avg_tx,
            "median_transaction": med_tx,
            "max_transaction": max_tx,
            "spending_variability": std_tx,
            "category_distribution": cat_dist,
            "top_categories": top_cats,
            "recurring_merchants": rec_merchants,
            "monthly_spending": monthly_spending,
            "spending_trend": spending_trend_desc,
            "category_concentration_pct": round(top2_sum, 2),
            "anomaly_frequency": anom_count
        }

        db_manager.save_behavior_profile(user_id, profile)
        logger.info(f"Computed behavior profile for user '{user_id}' ({len(tx_df)} transactions).")
        return profile
