import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from src.config import config
from src.utils import logger

CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "INR ",
    "EUR": "€",
    "GBP": "£",
    "AUD": "A$",
    "CAD": "C$",
    "JPY": "¥",
}

def format_currency_amount(amount: float, currency_str: str = "USD") -> str:
    """Format numeric amount using currency-specific symbol or code."""
    curr = str(currency_str).strip().upper() if currency_str else "USD"
    symbol = CURRENCY_SYMBOLS.get(curr, f"{curr} ")
    return f"{symbol}{amount:,.2f}"

class SpendingAnomalyDetector:
    """Detects unusual spending patterns using IsolationForest and statistical z-score thresholds."""

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.iso_forest = IsolationForest(contamination=contamination, random_state=config.random_seed)

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag unusual spending transactions and return enriched DataFrame."""
        if "amount" not in df.columns or len(df) < 5:
            logger.info("Insufficient rows or missing amount column for anomaly detection.")
            df_out = df.copy()
            df_out["is_unusual"] = False
            df_out["anomaly_score"] = 0.0
            df_out["anomaly_reason"] = "Insufficient data for anomaly detection"
            return df_out

        df_out = df.copy()
        amounts = pd.to_numeric(df_out["amount"], errors="coerce").abs().fillna(0.0)
        
        # Extract currency column if present
        currencies = df_out["currency"].astype(str).tolist() if "currency" in df_out.columns else ["USD"] * len(df_out)

        # 1. Statistical z-score computation
        mean_amt = amounts.mean()
        std_amt = amounts.std() if amounts.std() > 0 else 1.0
        z_scores = (amounts - mean_amt) / std_amt
        
        # 2. Isolation Forest model fitting
        X = amounts.values.reshape(-1, 1)
        self.iso_forest.fit(X)
        iso_preds = self.iso_forest.predict(X)  # -1 = anomaly, 1 = normal
        scores = -self.iso_forest.decision_function(X) # Higher = more unusual
        
        # Combine z-score (> 2.5) or Isolation Forest flag
        is_unusual = (iso_preds == -1) | (z_scores > 2.5)
        
        df_out["is_unusual"] = is_unusual
        df_out["anomaly_score"] = np.round(scores, 4)
        
        # Assign plain-language explanatory reason with proper currency formatting
        reasons = []
        for amt, z, un, curr in zip(amounts, z_scores, is_unusual, currencies):
            fmt_amt = format_currency_amount(amt, curr)
            if not un:
                reasons.append("Normal spending pattern")
            elif z > 3.0:
                reasons.append(f"Significantly high transaction amount ({fmt_amt}, >3x standard deviation above average)")
            elif z > 2.0:
                reasons.append(f"Unusual high amount relative to your average spending ({fmt_amt})")
            else:
                reasons.append(f"Unusual spending frequency / statistical outlier flagged by model ({fmt_amt})")
                
        df_out["anomaly_reason"] = reasons
        logger.info(f"Anomaly detection complete. Flagged {is_unusual.sum()} unusual transactions out of {len(df)}.")
        return df_out
