"""Comprehensive unit tests for v3-robustness capabilities.

Covers:
- Temperature scaling calibration (ECE_calibrated <= ECE_uncalibrated)
- Multi-user feedback isolation (User A feedback never leaks to User B)
- XGBoost label mapping index order regression check
- Offline LLM fallback when network / endpoint is unreachable
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.calibration import TemperatureScaler, compute_ece
from src.adaptive_nlp import AdaptiveNLPClassifier
from src.grounded_llm import GroundedFinancialAssistant
from src.database import db_manager


def test_temperature_scaling_calibration_ece():
    """Verify temperature scaling reduces or maintains ECE on miscalibrated predictions."""
    np.random.seed(42)
    n_samples = 500
    n_classes = 5

    # Generate synthetic true labels and overconfident logits
    y_true = np.random.randint(0, n_classes, size=n_samples)
    logits = np.random.randn(n_samples, n_classes) * 4.0  # High variance -> overconfidence
    
    # Uncalibrated softmax probabilities
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_raw = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ece_raw = compute_ece(y_true, probs_raw)

    scaler = TemperatureScaler().fit(logits, y_true)
    probs_calibrated = scaler.transform(logits)
    ece_calibrated = compute_ece(y_true, probs_calibrated)

    # Temperature scaling on overconfident predictions must reduce ECE
    assert ece_calibrated <= ece_raw + 1e-4, f"Calibrated ECE ({ece_calibrated:.4f}) should be <= raw ECE ({ece_raw:.4f})"
    assert scaler.temperature > 0.0


def test_user_feedback_isolation():
    """Verify that feedback from user_A never leaks into user_B's predictions."""
    adaptive = AdaptiveNLPClassifier()
    user_a = "test_user_isolation_A"
    user_b = "test_user_isolation_B"
    txn_desc = "Quick Mart Store #402"

    # User A records a correction: "Shopping & Retail"
    adaptive.record_user_correction(
        user_id=user_a,
        description=txn_desc,
        original_prediction="Food & Dining",
        corrected_category="Shopping & Retail"
    )

    # Predict for User A: should match personalized correction
    pred_a = adaptive.predict_personalized(user_id=user_a, transaction_description=txn_desc)
    assert pred_a["is_personalized"] is True
    assert pred_a["predicted_category"] == "Shopping & Retail"
    assert "User Exact Correction" in pred_a["personalization_source"]

    # Predict for User B: should NOT have User A's correction; must use global baseline
    pred_b = adaptive.predict_personalized(user_id=user_b, transaction_description=txn_desc)
    assert pred_b["is_personalized"] is False
    assert pred_b["personalization_source"] == "Global Linear SVM Baseline"
    assert pred_b["predicted_category"] != "Shopping & Retail" or pred_b["is_personalized"] is False


def test_xgboost_label_mapping_consistency():
    """Regression test ensuring XGBoost internal class indices strictly preserve label encoder order."""
    classes = ["Education", "Food & Dining", "Income", "Rent", "Transportation"]
    le = LabelEncoder().fit(classes)

    # Synthetic training data with 10 samples per class
    n_c = len(classes)
    X = np.repeat(np.eye(n_c), 10, axis=0)
    y = np.repeat(np.arange(n_c), 10)

    model = XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.5, random_state=42)
    model.fit(X, y)

    # 1. Verify model classes matches label encoder indices [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(model.classes_, np.arange(len(classes)))

    # 2. Verify model prediction maps to expected string categories in order
    preds = model.predict(np.eye(n_c))
    decoded = list(le.inverse_transform(preds))
    assert decoded == classes, f"Expected {classes}, got {decoded}"

    # 3. Verify predict_proba columns match le.classes_
    probas = model.predict_proba(np.eye(n_c))
    assert probas.shape == (n_c, n_c)
    assert np.argmax(probas, axis=1).tolist() == list(range(n_c))



def test_grounded_llm_unreachable_endpoint_fallback():
    """Verify that GroundedFinancialAssistant falls back gracefully when LLM endpoint is offline."""
    # Point to non-existent endpoint
    assistant = GroundedFinancialAssistant(
        api_key="dummy_key",
        base_url="http://127.0.0.1:59999/v1",
        model="llama3.2:3b"
    )

    df = pd.DataFrame([
        {"date": "2026-09-01", "transaction_description": "Salary Direct Deposit", "amount": 6000.0, "category": "Income"},
        {"date": "2026-09-05", "transaction_description": "Grocery Store Supermarket", "amount": -150.0, "category": "Food & Dining"}
    ])

    # Should not throw connection error; must fall back to deterministic template
    res = assistant.process_query("Where am I spending the most money?", df)
    assert res is not None
    assert res["is_llm_powered"] is False
    assert "Answer:" in res["answer"]
    assert res["tool_result"]["status"] == "success"
