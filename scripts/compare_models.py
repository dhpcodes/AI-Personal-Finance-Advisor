"""Comprehensive Multi-Model Benchmark Comparison Suite (Phase 3).

Evaluates and compares:
1. Production Baseline (models/final_model.joblib)
2. Variant B Candidate Linear SVM (models/variant_b_candidate.joblib)
3. LightGBM Classifier (models/lightgbm_transaction_classifier.joblib)
4. XGBoost Classifier (models/xgboost_transaction_classifier.joblib)
5. MiniLM Embeddings + Logistic Regression (models/minilm_lr_transaction_classifier.joblib)
6. DistilBERT Fine-Tuning (Colab predictions from results/distilbert/ or labeled NOT RUN)

Evaluations:
- Untouched Clean Test set (N=20,040)
- Noisy Test set (N=20,040, 8 noise conditions, seed 7)
- Typo Noise subset (N=2,492)
- Real Indian Bank Statements (N=105, model-only & model+rent rule)
- 95% Wilson Score Confidence Intervals on all accuracies
- Expected Calibration Error (ECE) on probability models
- McNemar test between top two models on Clean Test and 105 Indian sets
- Latency: Single-item and Batch (256), with and without preprocessing
- Target Indian Merchant Brands table (11 brands)

Outputs:
- results/final_metrics.csv (The Single Source of Truth)
- results/model_comparison.csv
- results/mcnemar_tests.json
- results/target_indian_merchants_all_models.csv
"""

import sys
import time
import random
import re
import json
import hashlib
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

from src.predict import TransactionPredictor
from src.preprocessing import TextPreprocessor
from src.model_comparison_suite import with_rent_rule, ece_score

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
DISTILBERT_DIR = RESULTS_DIR / "distilbert"

SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"

PRODUCTION_MODEL_PATH = MODELS_DIR / "final_model.joblib"
BACKUP_MODEL_PATH = MODELS_DIR / "final_model_pre_canonicalization_backup.joblib"
SVM_CANDIDATE_PATH = MODELS_DIR / "variant_b_candidate.joblib"
LIGHTGBM_PATH = MODELS_DIR / "lightgbm_transaction_classifier.joblib"
XGBOOST_PATH = MODELS_DIR / "xgboost_transaction_classifier.joblib"
MINILM_PATH = MODELS_DIR / "minilm_lr_transaction_classifier.joblib"

TEST_KINDS = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]

TARGET_BRANDS = [
    ("Amazon Pay", "UPI/998124019283/AMAZON PAY/amazon@apl", "Shopping & Retail"),
    ("Swiggy", "UPI/308942189012/SWIGGY/swiggy@icici", "Food & Dining"),
    ("Zomato", "UPI/489120938102/ZOMATO/zomato@hdfcbank", "Food & Dining"),
    ("Ola Cabs", "UPI/992019283102/OLA CABS/olacabs@icici", "Transportation"),
    ("Reliance Jio", "UPI/110293849102/RELIANCE JIO RECHARGE/jio@paytm", "Utilities & Services"),
    ("Blinkit", "BLINKIT COMMERCE PVT LTD", "Food & Dining"),
    ("Zepto", "ZEPTO 10 MIN GROCERY DELIVERY", "Food & Dining"),
    ("BigBasket", "BIGBASKET SUPERMARKET ORDER", "Food & Dining"),
    ("Flipkart", "IMPS/3948201/FLIPKART INTERNET/REF921", "Shopping & Retail"),
    ("Myntra", "POS 5241XXXX9021 MYNTRA DESIGNS BANGALORE", "Shopping & Retail"),
    ("Airtel", "BHARTI AIRTEL POSTPAID BILL PAYMENT", "Utilities & Services"),
]


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_single_noise(text: str, rng: random.Random, kind: str) -> str:
    ref = rng.randint(10**8, 10**12)
    vpa = re.sub(r"[^a-z]", "", text.lower())[:10] or "merchant"
    if kind == "upi":   return f"UPI/{ref}/{text.upper()}/{vpa}@okaxis"
    if kind == "neft":  return f"NEFT-{ref}-{text.upper()}"
    if kind == "pos":   return f"POS {rng.randint(1000,9999)}XXXX{rng.randint(1000,9999)} {text.upper()} MUMBAI"
    if kind == "imps":  return f"IMPS/{ref}/{text.upper()}/REF{rng.randint(100,999)}"
    if kind == "upper": return text.upper()
    if kind == "lower": return text.lower()
    if kind == "trunc": return text.upper()[: max(5, len(text) // 2)] + "..."
    if kind == "typo":
        s = list(text)
        if len(s) > 4:
            i = rng.randint(1, len(s) - 2)
            s[i], s[i + 1] = s[i + 1], s[i]
        return "".join(s)
    return text


def get_noise_type_own(desc: str) -> str:
    d = str(desc).upper()
    if "UPI" in d:  return "UPI"
    if "POS" in d:  return "POS"
    if "NEFT" in d: return "NEFT"
    if "IMPS" in d: return "IMPS"
    return "Clean_Indian"


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> str:
    """95% Wilson Score Interval for binomial proportion."""
    if n == 0:
        return "[0.0000, 0.0000]"
    z = 1.95996
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat)) / n + z**2 / (4 * n**2))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return f"[{lo*100:.2f}%, {hi*100:.2f}%]"


def mcnemar_test(y_true: list, preds_a: list, preds_b: list) -> Dict[str, Any]:
    """McNemar's test with continuity correction between two models."""
    arr_true = np.array(y_true)
    correct_a = (np.array(preds_a) == arr_true)
    correct_b = (np.array(preds_b) == arr_true)
    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n10 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    total_disc = n01 + n10
    if total_disc == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": 0, "n10": 0, "significant": False}
    stat = float((abs(n01 - n10) - 1.0)**2 / total_disc)
    p_val = float(1.0 - chi2.cdf(stat, df=1))
    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 5),
        "n01_b_better": n01,
        "n10_a_better": n10,
        "significant_p05": p_val < 0.05
    }


def measure_latencies(predictor: TransactionPredictor, raw_samples: list, warmup: int = 20) -> Dict[str, float]:
    """Benchmark single-call and batch latency, with and without preprocessing."""
    preprocessor = predictor.preprocessor
    cleaned_samples = [preprocessor.clean_text(s) for s in raw_samples]

    # Warmup
    for s in raw_samples[:warmup]:
        _ = predictor.predict_single(s)

    # 1. Single-item latency with preprocessing
    t_single_with = []
    for s in raw_samples[:100]:
        t0 = time.perf_counter()
        _ = predictor.predict_single(s)
        t_single_with.append(time.perf_counter() - t0)
    single_with_ms = float(np.mean(t_single_with) * 1000.0)

    # 2. Single-item latency without preprocessing (model/vectorizer forward pass only)
    t_single_without = []
    for c in cleaned_samples[:100]:
        t0 = time.perf_counter()
        if predictor.vectorizer is not None:
            vec = predictor.vectorizer.transform([c])
        elif predictor.encoder is not None:
            vec = predictor.encoder.encode([c], show_progress_bar=False, normalize_embeddings=True)
        if hasattr(predictor.model, "predict_proba"):
            _ = predictor.model.predict_proba(vec)
        else:
            _ = predictor.model.predict(vec)
        t_single_without.append(time.perf_counter() - t0)
    single_without_ms = float(np.mean(t_single_without) * 1000.0)

    # 3. Batch 256 latency with preprocessing
    batch_raw = (raw_samples * (256 // len(raw_samples) + 1))[:256]
    df_batch = pd.DataFrame({"transaction_description": batch_raw})
    t0 = time.perf_counter()
    _ = predictor.predict_batch(df_batch)
    batch_with_ms = float(((time.perf_counter() - t0) / 256) * 1000.0)

    # 4. Batch 256 latency without preprocessing
    batch_clean = (cleaned_samples * (256 // len(cleaned_samples) + 1))[:256]
    t0 = time.perf_counter()
    if predictor.vectorizer is not None:
        X_b = predictor.vectorizer.transform(batch_clean)
    elif predictor.encoder is not None:
        X_b = predictor.encoder.encode(batch_clean, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    if hasattr(predictor.model, "predict_proba"):
        _ = predictor.model.predict_proba(X_b)
    else:
        _ = predictor.model.predict(X_b)
    batch_without_ms = float(((time.perf_counter() - t0) / 256) * 1000.0)

    return {
        "single_with_prep_ms": round(single_with_ms, 3),
        "single_no_prep_ms": round(single_without_ms, 3),
        "batch256_with_prep_ms": round(batch_with_ms, 3),
        "batch256_no_prep_ms": round(batch_without_ms, 3)
    }


def get_prediction_probs(predictor: TransactionPredictor, cleaned_texts: list) -> np.ndarray:
    """Extract probability matrix for ECE calculation."""
    if predictor.vectorizer is not None:
        X_vec = predictor.vectorizer.transform(cleaned_texts)
    elif predictor.encoder is not None:
        X_vec = predictor.encoder.encode(cleaned_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    else:
        raise ValueError("Missing feature extractor.")

    if hasattr(predictor.model, "predict_proba"):
        return predictor.model.predict_proba(X_vec)
    elif hasattr(predictor.model, "decision_function"):
        df_scores = predictor.model.decision_function(X_vec)
        exp_s = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)
    else:
        preds = predictor.model.predict(X_vec)
        probs = np.zeros((len(preds), len(predictor.classes)))
        for idx, p in enumerate(preds):
            probs[idx, p] = 1.0
        return probs


def main():
    print("=" * 70)
    print("AI PERSONAL FINANCE ADVISOR: PHASE 3 FAIR COMPARISON SUITE")
    print("=" * 70)

    # 1. Safety verification
    prod_hash_before = get_file_sha256(PRODUCTION_MODEL_PATH)
    backup_hash_before = get_file_sha256(BACKUP_MODEL_PATH)
    print(f"[Safety Check] Production Model SHA256: {prod_hash_before}")
    print(f"[Safety Check] Backup Model SHA256:     {backup_hash_before}")
    assert prod_hash_before == backup_hash_before, "Production hash mismatch!"

    # 2. Predictors dictionary
    print("\n[Step 1] Loading all available model predictors...")
    predictors = {
        "Production_Baseline": TransactionPredictor(model_path=PRODUCTION_MODEL_PATH),
        "Candidate_Linear_SVM": TransactionPredictor(model_path=SVM_CANDIDATE_PATH),
        "LightGBM": TransactionPredictor(model_path=LIGHTGBM_PATH),
        "XGBoost": TransactionPredictor(model_path=XGBOOST_PATH),
        "MiniLM_LogReg": TransactionPredictor(model_path=MINILM_PATH),
    }

    # 3. Load Datasets
    print("\n[Step 2] Loading test partitions and generating noise...")
    df_split = pd.read_csv(SPLIT_PATH)
    test_df = df_split[df_split.split == "test"].copy()
    val_df = df_split[df_split.split == "val"].copy()

    test_clean_descs = test_df.transaction_description.tolist()
    y_test = test_df.category.tolist()

    # Deterministic noise on test set (seed 7)
    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
    test_noisy_descs = [
        apply_single_noise(t, r_te, nt) for t, nt in zip(test_clean_descs, test_noise_types)
    ]
    test_df["noise_type"] = test_noise_types
    test_df["noisy_description"] = test_noisy_descs

    # Real Indian statements
    df_own = pd.read_csv(OWN_TEST_PATH)
    y_own = df_own.category.tolist()

    # Label encoder for numerical indices (for ECE calculation)
    le = LabelEncoder().fit(y_test)
    y_test_indices = le.transform(y_test)

    # 4. Evaluation Loop
    final_rows = []
    model_clean_preds = {}
    model_own_preds = {}

    preprocessor = TextPreprocessor(use_canonicalization=True)
    test_clean_preprocessed = [preprocessor.clean_text(t) for t in test_clean_descs]

    for name, predictor in predictors.items():
        print(f"\nEvaluating {name}...")
        artifact = joblib.load(predictor.model_path)
        art_size_mb = round(predictor.model_path.stat().st_size / (1024 * 1024), 2)
        fit_time = artifact.get("metrics", {}).get("fit_time_sec", None)
        val_acc = artifact.get("metrics", {}).get("val_accuracy", None)
        val_f1 = artifact.get("metrics", {}).get("val_macro_f1", None)

        # 4a. Clean Test Predictions
        res_clean = predictor.predict_batch(pd.DataFrame({"transaction_description": test_clean_descs}))
        clean_preds = res_clean["predicted_category"].tolist()
        model_clean_preds[name] = clean_preds
        k_clean = sum(p == y for p, y in zip(clean_preds, y_test))
        clean_acc = round(accuracy_score(y_test, clean_preds), 4)
        clean_ci = wilson_ci(k_clean, len(y_test))
        clean_macro_f1 = round(f1_score(y_test, clean_preds, average="macro", zero_division=0), 4)
        clean_weighted_f1 = round(f1_score(y_test, clean_preds, average="weighted", zero_division=0), 4)

        # 4b. Noisy Test Predictions
        res_noisy = predictor.predict_batch(pd.DataFrame({"transaction_description": test_noisy_descs}))
        noisy_preds = res_noisy["predicted_category"].tolist()
        k_noisy = sum(p == y for p, y in zip(noisy_preds, y_test))
        noisy_acc = round(accuracy_score(y_test, noisy_preds), 4)
        noisy_ci = wilson_ci(k_noisy, len(y_test))
        noisy_macro_f1 = round(f1_score(y_test, noisy_preds, average="macro", zero_division=0), 4)
        noisy_weighted_f1 = round(f1_score(y_test, noisy_preds, average="weighted", zero_division=0), 4)

        # 4c. Typo Subset
        test_df[f"pred_noisy_{name}"] = noisy_preds
        typo_mask = test_df.noise_type == "typo"
        typo_true = test_df.loc[typo_mask, "category"].tolist()
        typo_pred = test_df.loc[typo_mask, f"pred_noisy_{name}"].tolist()
        k_typo = sum(p == y for p, y in zip(typo_pred, typo_true))
        typo_acc = round(accuracy_score(typo_true, typo_pred), 4)
        typo_ci = wilson_ci(k_typo, len(typo_true))

        # 4d. Indian Statements (N=105)
        res_own = predictor.predict_batch(df_own)
        own_preds = res_own["predicted_category"].tolist()
        model_own_preds[name] = own_preds
        k_own_model = sum(p == y for p, y in zip(own_preds, y_own))
        own_model_acc = round(accuracy_score(y_own, own_preds), 4)
        own_model_ci = wilson_ci(k_own_model, len(y_own))

        own_rule_preds = with_rent_rule(df_own.transaction_description, pd.Series(own_preds))
        k_own_rule = sum(p == y for p, y in zip(own_rule_preds, y_own))
        own_rule_acc = round(accuracy_score(y_own, own_rule_preds), 4)
        own_rule_ci = wilson_ci(k_own_rule, len(y_own))
        own_rule_f1 = round(f1_score(y_own, own_rule_preds, average="macro", zero_division=0), 4)

        # 4e. Expected Calibration Error (ECE)
        try:
            probs = get_prediction_probs(predictor, test_clean_preprocessed)
            ece_val = ece_score(y_test_indices, probs, bins=15)
        except Exception as e:
            print(f"  Warning: ECE calculation failed for {name}: {e}")
            ece_val = None

        # 4f. Latencies
        lats = measure_latencies(predictor, test_clean_descs[:200])

        final_rows.append({
            "Model": name,
            "Family": (
                "Linear (Legacy)" if name == "Production_Baseline" else
                "Linear (SVM)" if name == "Candidate_Linear_SVM" else
                "Gradient Boosted Trees" if name in ["LightGBM", "XGBoost"] else
                "Dense Sentence Embeddings" if name == "MiniLM_LogReg" else "Transformer"
            ),
            "Status": "MEASURED",
            "Validation_Accuracy": val_acc,
            "Validation_Macro_F1": val_f1,
            "Clean_Test_Accuracy": clean_acc,
            "Clean_Test_95_CI": clean_ci,
            "Clean_Test_Macro_F1": clean_macro_f1,
            "Clean_Test_Weighted_F1": clean_weighted_f1,
            "Noisy_Test_Accuracy": noisy_acc,
            "Noisy_Test_95_CI": noisy_ci,
            "Noisy_Test_Macro_F1": noisy_macro_f1,
            "Noisy_Test_Weighted_F1": noisy_weighted_f1,
            "Typo_Accuracy": typo_acc,
            "Typo_95_CI": typo_ci,
            "Indian_Model_Acc": own_model_acc,
            "Indian_Model_95_CI": own_model_ci,
            "Indian_Model_Plus_Rent_Acc": own_rule_acc,
            "Indian_Model_Plus_Rent_95_CI": own_rule_ci,
            "Indian_Statement_Macro_F1": own_rule_f1,
            "ECE": ece_val,
            "Single_Latency_With_Prep_ms": lats["single_with_prep_ms"],
            "Single_Latency_No_Prep_ms": lats["single_no_prep_ms"],
            "Batch256_Latency_With_Prep_ms": lats["batch256_with_prep_ms"],
            "Batch256_Latency_No_Prep_ms": lats["batch256_no_prep_ms"],
            "Training_Time_Sec": fit_time,
            "Artifact_Size_MB": art_size_mb
        })

    # 5. Check if DistilBERT predictions exist
    distilbert_clean_csv = DISTILBERT_DIR / "distilbert_preds_clean_test.csv"
    if distilbert_clean_csv.exists():
        print("\nLoading DistilBERT predictions from results/distilbert/...")
        df_d_clean = pd.read_csv(DISTILBERT_DIR / "distilbert_preds_clean_test.csv")
        df_d_noisy = pd.read_csv(DISTILBERT_DIR / "distilbert_preds_noisy_test.csv")
        df_d_typo = pd.read_csv(DISTILBERT_DIR / "distilbert_preds_typo_test.csv")
        df_d_own = pd.read_csv(DISTILBERT_DIR / "distilbert_preds_own_test_indian.csv")

        d_clean_preds = df_d_clean["predicted_category"].tolist()
        model_clean_preds["DistilBERT"] = d_clean_preds
        k_d_clean = sum(p == y for p, y in zip(d_clean_preds, y_test))
        d_clean_acc = round(accuracy_score(y_test, d_clean_preds), 4)

        d_noisy_preds = df_d_noisy["predicted_category"].tolist()
        k_d_noisy = sum(p == y for p, y in zip(d_noisy_preds, y_test))
        d_noisy_acc = round(accuracy_score(y_test, d_noisy_preds), 4)

        d_typo_preds = df_d_typo["predicted_category"].tolist()
        d_typo_true = df_d_typo["true_category"].tolist()
        k_d_typo = sum(p == y for p, y in zip(d_typo_preds, d_typo_true))
        d_typo_acc = round(accuracy_score(d_typo_true, d_typo_preds), 4)

        d_own_preds = df_d_own["predicted_category"].tolist()
        model_own_preds["DistilBERT"] = d_own_preds
        k_d_own = sum(p == y for p, y in zip(d_own_preds, y_own))
        d_own_acc = round(accuracy_score(y_own, d_own_preds), 4)

        d_rule_preds = with_rent_rule(df_d_own["transaction_description"], pd.Series(d_own_preds))
        k_d_rule = sum(p == y for p, y in zip(d_rule_preds, y_own))
        d_rule_acc = round(accuracy_score(y_own, d_rule_preds), 4)
        d_rule_f1 = round(f1_score(y_own, d_rule_preds, average="macro", zero_division=0), 4)

        final_rows.append({
            "Model": "DistilBERT",
            "Family": "Fine-Tuned Transformer",
            "Status": "MEASURED (Colab T4)",
            "Validation_Accuracy": None,
            "Validation_Macro_F1": None,
            "Clean_Test_Accuracy": d_clean_acc,
            "Clean_Test_95_CI": wilson_ci(k_d_clean, len(y_test)),
            "Clean_Test_Macro_F1": round(f1_score(y_test, d_clean_preds, average="macro", zero_division=0), 4),
            "Clean_Test_Weighted_F1": round(f1_score(y_test, d_clean_preds, average="weighted", zero_division=0), 4),
            "Noisy_Test_Accuracy": d_noisy_acc,
            "Noisy_Test_95_CI": wilson_ci(k_d_noisy, len(y_test)),
            "Noisy_Test_Macro_F1": round(f1_score(y_test, d_noisy_preds, average="macro", zero_division=0), 4),
            "Noisy_Test_Weighted_F1": round(f1_score(y_test, d_noisy_preds, average="weighted", zero_division=0), 4),
            "Typo_Accuracy": d_typo_acc,
            "Typo_95_CI": wilson_ci(k_d_typo, len(d_typo_true)),
            "Indian_Model_Acc": d_own_acc,
            "Indian_Model_95_CI": wilson_ci(k_d_own, len(y_own)),
            "Indian_Model_Plus_Rent_Acc": d_rule_acc,
            "Indian_Model_Plus_Rent_95_CI": wilson_ci(k_d_rule, len(y_own)),
            "Indian_Statement_Macro_F1": d_rule_f1,
            "ECE": None,
            "Single_Latency_With_Prep_ms": None,
            "Single_Latency_No_Prep_ms": None,
            "Batch256_Latency_With_Prep_ms": None,
            "Batch256_Latency_No_Prep_ms": None,
            "Training_Time_Sec": None,
            "Artifact_Size_MB": None
        })
    else:
        print("\nDistilBERT predictions not found in results/distilbert/. Recording as NOT RUN.")
        final_rows.append({
            "Model": "DistilBERT",
            "Family": "Fine-Tuned Transformer",
            "Status": "NOT RUN (Planned on Colab T4)",
            "Validation_Accuracy": None,
            "Validation_Macro_F1": None,
            "Clean_Test_Accuracy": None,
            "Clean_Test_95_CI": None,
            "Clean_Test_Macro_F1": None,
            "Clean_Test_Weighted_F1": None,
            "Noisy_Test_Accuracy": None,
            "Noisy_Test_95_CI": None,
            "Noisy_Test_Macro_F1": None,
            "Noisy_Test_Weighted_F1": None,
            "Typo_Accuracy": None,
            "Typo_95_CI": None,
            "Indian_Model_Acc": None,
            "Indian_Model_95_CI": None,
            "Indian_Model_Plus_Rent_Acc": None,
            "Indian_Model_Plus_Rent_95_CI": None,
            "Indian_Statement_Macro_F1": None,
            "ECE": None,
            "Single_Latency_With_Prep_ms": None,
            "Single_Latency_No_Prep_ms": None,
            "Batch256_Latency_With_Prep_ms": None,
            "Batch256_Latency_No_Prep_ms": None,
            "Training_Time_Sec": None,
            "Artifact_Size_MB": None
        })

    # Save final_metrics.csv
    df_final = pd.DataFrame(final_rows)
    final_csv_path = RESULTS_DIR / "final_metrics.csv"
    df_final.to_csv(final_csv_path, index=False)
    print(f"\n[SAVED] Single Source of Truth metrics saved to {final_csv_path}.")

    # 6. McNemar Tests between top two models
    # On clean test: top two are Candidate_Linear_SVM (0.9900) vs LightGBM (0.9772)
    mcnemar_results = {}
    if "Candidate_Linear_SVM" in model_clean_preds and "LightGBM" in model_clean_preds:
        mc_clean = mcnemar_test(
            y_test,
            model_clean_preds["Candidate_Linear_SVM"],
            model_clean_preds["LightGBM"]
        )
        mcnemar_results["Clean_Test_SVM_vs_LightGBM"] = {
            "model_a": "Candidate_Linear_SVM",
            "model_b": "LightGBM",
            "n_samples": len(y_test),
            **mc_clean
        }
        print(f"\n[McNemar Test - Clean Test] Linear SVM vs LightGBM: chi2={mc_clean['statistic']}, p={mc_clean['p_value']}")

    # On Indian 105 set: top two are Candidate_Linear_SVM vs LightGBM (or MiniLM)
    if "Candidate_Linear_SVM" in model_own_preds and "MiniLM_LogReg" in model_own_preds:
        mc_own_svm_minilm = mcnemar_test(
            y_own,
            model_own_preds["Candidate_Linear_SVM"],
            model_own_preds["MiniLM_LogReg"]
        )
        mcnemar_results["Indian_105_SVM_vs_MiniLM"] = {
            "model_a": "Candidate_Linear_SVM",
            "model_b": "MiniLM_LogReg",
            "n_samples": len(y_own),
            **mc_own_svm_minilm
        }
        print(f"[McNemar Test - Indian 105] Linear SVM vs MiniLM_LogReg: chi2={mc_own_svm_minilm['statistic']}, p={mc_own_svm_minilm['p_value']}")

    if "LightGBM" in model_own_preds and "MiniLM_LogReg" in model_own_preds:
        mc_own_lgb_minilm = mcnemar_test(
            y_own,
            model_own_preds["LightGBM"],
            model_own_preds["MiniLM_LogReg"]
        )
        mcnemar_results["Indian_105_LightGBM_vs_MiniLM"] = {
            "model_a": "LightGBM",
            "model_b": "MiniLM_LogReg",
            "n_samples": len(y_own),
            **mc_own_lgb_minilm
        }
        print(f"[McNemar Test - Indian 105] LightGBM vs MiniLM_LogReg: chi2={mc_own_lgb_minilm['statistic']}, p={mc_own_lgb_minilm['p_value']}")

    mcnemar_path = RESULTS_DIR / "mcnemar_tests.json"
    with open(mcnemar_path, "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    print(f"[SAVED] McNemar test results saved to {mcnemar_path}.")

    # 7. Evaluate 11 Target Brands
    print("\n[Step 5] Evaluating 11 Target Brands across models...")
    brand_rows = []
    for brand, text, expected in TARGET_BRANDS:
        row = {"Brand": brand, "Input_Description": text, "Expected_Category": expected}
        for m_name, pred_obj in predictors.items():
            r = pred_obj.predict_single(text)
            p_cat = r["predicted_category"]
            c_val = r["confidence"]
            mark = "PASS" if p_cat == expected else "FAIL"
            row[f"{m_name}_Pred"] = p_cat
            row[f"{m_name}_Conf"] = c_val
            row[f"{m_name}_Status"] = mark
        brand_rows.append(row)

    df_brands = pd.DataFrame(brand_rows)
    brands_path = RESULTS_DIR / "target_indian_merchants_all_models.csv"
    df_brands.to_csv(brands_path, index=False)
    print(f"[SAVED] Target brands comparison saved to {brands_path}.")

    print("\n" + "=" * 70)
    print("PHASE 3 FAIR COMPARISON BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
