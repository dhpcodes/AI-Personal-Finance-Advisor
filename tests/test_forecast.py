import pytest
import numpy as np
import pandas as pd
from src.forecast import SyntheticDataGenerator, ForecastMetrics, BacktestEngine, PersonalizedSpendingForecaster

def test_synthetic_data_generator():
    df = SyntheticDataGenerator.generate_persona_transactions(persona="salaried_tech", num_months=12, seed=42)
    assert len(df) > 50
    assert "is_synthetic" in df.columns
    assert df["is_synthetic"].all()
    assert "SYNTHETIC" in df["dataset_note"].iloc[0]

def test_forecast_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    # Errors: |10|, |10|, |0| -> MAE = 20 / 3 = 6.6667
    assert round(ForecastMetrics.mae(y_true, y_pred), 2) == 6.67
    assert round(ForecastMetrics.rmse(y_true, y_pred), 2) == 8.16

    # sMAPE
    smape = ForecastMetrics.smape(y_true, y_pred)
    assert smape > 0.0

def test_rolling_origin_backtest():
    # 12 months synthetic monthly series
    months = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m")
    values = [40000, 42000, 41000, 43000, 45000, 44000, 46000, 47000, 48000, 60000, 52000, 50000]
    series = pd.Series(values, index=months)

    report = BacktestEngine.run_rolling_origin_backtest(series, min_train_length=6)
    assert len(report) == 3 # 3 baselines
    assert "Last-Month Naive" in report["Forecasting_Model"].values
    assert "Moving-Average (3-Month SMA)" in report["Forecasting_Model"].values

def test_user_forecast_insufficient_data():
    # Only 2 transactions / 1 month -> insufficient
    df = pd.DataFrame([
        {"date": "2026-08-01", "amount": -500.0, "category": "Food & Dining"},
        {"date": "2026-08-15", "amount": -700.0, "category": "Food & Dining"}
    ])
    res = PersonalizedSpendingForecaster.forecast_user_expenses(df)
    assert res["status"] == "insufficient_data"

def test_user_forecast_sufficient_data():
    # 4 months of expenses
    df = pd.DataFrame([
        {"date": "2026-05-01", "amount": -30000.0, "category": "Food & Dining"},
        {"date": "2026-06-01", "amount": -32000.0, "category": "Food & Dining"},
        {"date": "2026-07-01", "amount": -31000.0, "category": "Food & Dining"},
        {"date": "2026-08-01", "amount": -33000.0, "category": "Food & Dining"},
    ])
    res = PersonalizedSpendingForecaster.forecast_user_expenses(df)
    assert res["status"] == "success"
    assert res["point_forecast"] > 0
    assert res["lower_bound"] <= res["point_forecast"] <= res["upper_bound"]
    assert "Food & Dining" in res["category_forecast"]
