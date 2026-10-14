"""Automated test suite for LightGBM and XGBoost Transaction Classifiers.

Verifies:
1. Artifact integrity and required dictionary keys.
2. Label encoder classes and valid taxonomy.
3. Feature extractor vocabulary and sparse matrix dimensions.
4. Seamless integration with TransactionPredictor.
5. Handling of empty, malformed, and unknown merchant inputs.
6. Repeated prediction consistency (determinism).
7. Strict preservation and invariance of production model artifacts.
"""

import hashlib
import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.predict import TransactionPredictor
from src.preprocessing import TextPreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
LGBM_MODEL_PATH = MODELS_DIR / "lightgbm_transaction_classifier.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_transaction_classifier.joblib"
PROD_MODEL_PATH = MODELS_DIR / "final_model.joblib"
BACKUP_MODEL_PATH = MODELS_DIR / "final_model_pre_canonicalization_backup.joblib"

EXPECTED_PROD_HASH = "fd898d98b9f6ec03902d825082d69a461d32b3318bca47b98f45e8e8dfc11195"


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def lgbm_predictor():
    return TransactionPredictor(model_path=LGBM_MODEL_PATH)


@pytest.fixture(scope="module")
def xgb_predictor():
    return TransactionPredictor(model_path=XGB_MODEL_PATH)


# --- 1. ARTIFACT INTEGRITY & STRUCTURE ---

def test_production_and_backup_hashes_unmodified():
    """Verify production and backup model artifacts have not been modified."""
    assert PROD_MODEL_PATH.exists(), "Production model artifact missing!"
    assert BACKUP_MODEL_PATH.exists(), "Backup model artifact missing!"
    
    prod_hash = get_file_sha256(PROD_MODEL_PATH)
    backup_hash = get_file_sha256(BACKUP_MODEL_PATH)
    
    assert prod_hash == EXPECTED_PROD_HASH, f"Production hash changed: {prod_hash}"
    assert backup_hash == EXPECTED_PROD_HASH, f"Backup hash changed: {backup_hash}"


@pytest.mark.parametrize("model_path,expected_version", [
    (LGBM_MODEL_PATH, "1.0.0-lightgbm"),
    (XGB_MODEL_PATH, "1.0.0-xgboost"),
])
def test_boosted_artifacts_structure(model_path, expected_version):
    """Verify boosted model artifacts contain all required keys and valid schema."""
    assert model_path.exists(), f"Artifact missing at {model_path}"
    art = joblib.load(model_path)
    
    required_keys = [
        "version", "model_name", "vectorizer", "label_encoder",
        "model", "classes", "metrics", "preprocessing_metadata"
    ]
    for k in required_keys:
        assert k in art, f"Missing required key '{k}' in {model_path}"
        
    assert art["version"] == expected_version
    assert len(art["classes"]) == 10
    assert hasattr(art["model"], "predict_proba")
    assert art["vectorizer"].analyzer == "char_wb"
    assert len(art["vectorizer"].vocabulary_) == 3000


# --- 2. PREDICTION INTERFACE & METRICS COMPATIBILITY ---

@pytest.mark.parametrize("fixture_name", ["lgbm_predictor", "xgb_predictor"])
def test_predictor_predict_single_schema(fixture_name, request):
    """Verify predict_single returns valid category, confidence, and top-k probabilities."""
    predictor = request.getfixturevalue(fixture_name)
    res = predictor.predict_single("UBER TRIP MUMBAI CAB")
    
    assert "predicted_category" in res
    assert "confidence" in res
    assert "top_k" in res
    assert res["predicted_category"] in predictor.classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 3
    assert res["top_k"][0]["category"] == res["predicted_category"]


@pytest.mark.parametrize("fixture_name", ["lgbm_predictor", "xgb_predictor"])
def test_predictor_predict_batch_schema(fixture_name, request):
    """Verify predict_batch operates cleanly on pandas DataFrames."""
    predictor = request.getfixturevalue(fixture_name)
    df_in = pd.DataFrame({
        "transaction_description": [
            "NETFLIX SUBSCRIPTION GURGAON",
            "SHELL OIL PETROL PUMP BANGALORE",
            "PHARMACY MEDICAL STORE"
        ]
    })
    df_out = predictor.predict_batch(df_in)
    
    assert len(df_out) == 3
    assert "predicted_category" in df_out.columns
    assert "confidence" in df_out.columns
    for cat in df_out["predicted_category"]:
        assert cat in predictor.classes


# --- 3. EDGE CASES & ROBUSTNESS ---

@pytest.mark.parametrize("fixture_name", ["lgbm_predictor", "xgb_predictor"])
def test_empty_and_whitespace_handling(fixture_name, request):
    """Verify empty and whitespace-only descriptions are safely handled."""
    predictor = request.getfixturevalue(fixture_name)
    
    for empty_input in ["", "   ", "\t\n  "]:
        res = predictor.predict_single(empty_input)
        assert res["predicted_category"] == "Unknown"
        assert res["confidence"] == 0.0
        assert res["top_k"] == []


@pytest.mark.parametrize("fixture_name", ["lgbm_predictor", "xgb_predictor"])
def test_unknown_merchant_handling(fixture_name, request):
    """Verify completely unknown or gibberish input produces a valid category without crashing."""
    predictor = request.getfixturevalue(fixture_name)
    res = predictor.predict_single("XYZQWERTYUNKNOWN987654321")
    
    assert res["predicted_category"] in predictor.classes
    assert 0.0 <= res["confidence"] <= 1.0


@pytest.mark.parametrize("fixture_name", ["lgbm_predictor", "xgb_predictor"])
def test_prediction_determinism(fixture_name, request):
    """Verify repeated predictions on identical text return identical probabilities."""
    predictor = request.getfixturevalue(fixture_name)
    text = "UPI/998124019283/AMAZON PAY/amazon@apl"
    
    res1 = predictor.predict_single(text)
    res2 = predictor.predict_single(text)
    
    assert res1["predicted_category"] == res2["predicted_category"]
    assert res1["confidence"] == res2["confidence"]
    assert res1["top_k"] == res2["top_k"]
