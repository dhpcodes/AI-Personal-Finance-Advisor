"""Tests for Experimental Variant B Candidate Model in Isolation."""

import hashlib
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.predict import TransactionPredictor

CANDIDATE_MODEL_PATH = Path("models/variant_b_candidate.joblib")
PRODUCTION_MODEL_PATH = Path("models/final_model.joblib")
BACKUP_MODEL_PATH = Path("models/final_model_pre_canonicalization_backup.joblib")

EXPECTED_PROD_HASH = "fd898d98b9f6ec03902d825082d69a461d32b3318bca47b98f45e8e8dfc11195"


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def candidate_predictor():
    assert CANDIDATE_MODEL_PATH.exists(), f"Candidate model not found at {CANDIDATE_MODEL_PATH}"
    return TransactionPredictor(model_path=CANDIDATE_MODEL_PATH)


def test_production_and_backup_hashes_strictly_unmodified():
    """Verify production and backup model artifacts have unchanged SHA256 hashes."""
    assert PRODUCTION_MODEL_PATH.exists()
    assert BACKUP_MODEL_PATH.exists()
    
    prod_hash = get_file_sha256(PRODUCTION_MODEL_PATH)
    backup_hash = get_file_sha256(BACKUP_MODEL_PATH)
    
    assert prod_hash == EXPECTED_PROD_HASH, f"Production model hash changed: {prod_hash}"
    assert backup_hash == EXPECTED_PROD_HASH, f"Backup model hash changed: {backup_hash}"
    assert prod_hash == backup_hash, "Production and backup model hashes differ!"


def test_candidate_artifact_structure():
    """Verify candidate artifact contains all required metadata and components."""
    import joblib
    artifact = joblib.load(CANDIDATE_MODEL_PATH)
    required_keys = [
        "version", "model_name", "vectorizer", "label_encoder",
        "model", "classes", "metrics", "preprocessing_metadata"
    ]
    for k in required_keys:
        assert k in artifact, f"Missing required key '{k}' in candidate artifact"
        
    assert "variant_b" in artifact["version"]
    assert len(artifact["classes"]) == 10
    assert hasattr(artifact["model"], "predict_proba")
    assert artifact["vectorizer"].analyzer == "char_wb"


def test_candidate_predictor_interface_compatibility(candidate_predictor):
    """Verify candidate model integrates seamlessly with TransactionPredictor."""
    res = candidate_predictor.predict_single("Shell Gas Station Fuel")
    assert "predicted_category" in res
    assert "confidence" in res
    assert "top_k" in res
    assert res["predicted_category"] == "Transportation"
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 3

    # Batch prediction
    df = pd.DataFrame({
        "transaction_description": ["Monthly Salary Direct Deposit", "Shell Gas Station Fuel"]
    })
    res_df = candidate_predictor.predict_batch(df)
    assert "predicted_category" in res_df.columns
    assert "confidence" in res_df.columns
    assert len(res_df) == 2


def test_candidate_amazon_pay_resolution(candidate_predictor):
    """Verify candidate resolves Amazon Pay to Shopping & Retail instead of Income."""
    res = candidate_predictor.predict_single("UPI/998124019283/AMAZON PAY/amazon@apl")
    assert res["predicted_category"] == "Shopping & Retail", (
        f"Expected Shopping & Retail for Amazon Pay, got {res['predicted_category']}"
    )
    assert res["confidence"] > 0.60


def test_candidate_typo_resilience(candidate_predictor):
    """Verify candidate character n-grams maintain classification under typo perturbation."""
    res_clean = candidate_predictor.predict_single("NETFLIX ENTERTAINMENT SUBSCRIPTION")
    res_typo = candidate_predictor.predict_single("NETLFIX ENTERTAINMENT SUBSCRIPTION")
    
    assert res_clean["predicted_category"] == "Entertainment & Recreation"
    assert res_typo["predicted_category"] == "Entertainment & Recreation"
    assert res_typo["confidence"] > 0.80
