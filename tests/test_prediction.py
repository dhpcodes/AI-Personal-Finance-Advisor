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

def test_rent_rule_word_boundaries_and_false_positives():
    from src.model_comparison_suite import with_rent_rule

    # True positives (should map to Rent)
    rent_texts = [
        "pg rent",
        "flat rent",
        "landlord transfer",
        "House rent payment April",
        "Monthly PG accommodation fee",
        "Transfer to landlord for flat 302",
        "Office lease agreement payment",
        "Housing society maintenance charge"
    ]
    model_labels = ["Utilities & Services"] * len(rent_texts)
    results = with_rent_rule(rent_texts, model_labels)
    assert all(r == "Rent" for r in results), f"Expected all Rent, got {results}"

    # False positives (should NOT map to Rent)
    non_rent_texts = [
        "page",
        "PG&E",
        "mortgage",
        "PG&E bill utility payment",
        "PG & E electric and gas service",
        "Page turner bookstore purchase",
        "Home mortgage payment Wells Fargo",
        "Engagement ring shop",
        "Upgraded storage subscription"
    ]
    model_labels = ["Utilities & Services"] * len(non_rent_texts)
    results = with_rent_rule(non_rent_texts, model_labels)
    assert all(r == "Utilities & Services" for r in results), f"Expected none to be Rent, got {results}"
