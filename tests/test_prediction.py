import pytest
import pandas as pd
from src.predict import TransactionPredictor
from src.explainability import ModelExplainer

@pytest.fixture
def predictor():
    return TransactionPredictor()

def test_prediction_single(predictor):
    res = predictor.predict_single("Starbucks Coffee #9921")
    assert "predicted_category" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) > 0

def test_prediction_batch(predictor):
    df = pd.DataFrame({
        "transaction_description": ["Monthly Salary Direct Deposit", "Shell Gas Station Fuel"]
    })
    res_df = predictor.predict_batch(df)
    assert "predicted_category" in res_df.columns
    assert "confidence" in res_df.columns
    assert len(res_df) == 2

def test_explainability(predictor):
    explainer = ModelExplainer(predictor)
    exp = explainer.explain_prediction("UCLA Medical #7731 Health Center")
    assert "explanation_summary" in exp
    assert "important_features" in exp
