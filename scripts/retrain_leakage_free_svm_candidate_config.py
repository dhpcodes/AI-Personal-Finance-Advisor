"""Retrain Leakage-Free SVM with Exact Original Candidate Configuration (Task 6).

Comparison Investigation:
- Original Candidate SVM used:
  1. max_features=20,000 with ngram_range=(2, 5)
  2. 35% synthetic training noise augmentation (seed=42)
  3. LinearSVC(C=10.0, max_iter=2000) with CalibratedClassifierCV(method="sigmoid")
- The previous leakage-free run used:
  1. max_features=3,000 with ngram_range=(2, 4)
  2. 0% training noise augmentation
  3. LinearSVC(C=1.0) with CalibratedClassifierCV(cv=3)

This script retrains the leakage-free model (reduced seed, N=72,838) using the EXACT
original candidate configuration (20k features, char_wb 2-5, C=10.0, 35% noise).
Saves to models/variant_b_candidate_leakage_free_20k.joblib.
"""

import sys
import time
import random
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from src.preprocessing import TextPreprocessor
from src.model_comparison_suite import with_rent_rule

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
CLEAN_SEED_PATH = DATA_DIR / "indian_merchants_train_seed_leakage_free.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"

OUTPUT_MODEL_PATH = MODELS_DIR / "variant_b_candidate_leakage_free_20k.joblib"

TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]

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

def main():
    print("=" * 70)
    print("TASK 6: RETRAINING LEAKAGE-FREE SVM WITH ORIGINAL CONFIGURATION")
    print("=" * 70)

    # 1. Load data
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_clean_seed = pd.read_csv(CLEAN_SEED_PATH)

    df_train_clean = pd.concat([train_base, df_clean_seed], ignore_index=True)
    print(f"Total Reduced Training Set: {len(df_train_clean):,} rows")

    le = LabelEncoder().fit(df_train_clean["category"])
    classes = list(le.classes_)
    y_tr = le.transform(df_train_clean["category"])
    y_va = le.transform(val_df["category"])
    y_te = le.transform(test_df["category"])

    # 2. Apply 35% synthetic training noise (seed=42) matching original candidate
    print("\n[Step 1] Applying 35% training noise augmentation (seed=42)...")
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(df_train_clean)) < 0.35
    train_noisy = [
        apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) if is_n else t
        for is_n, t in zip(m_tr, df_train_clean["transaction_description"])
    ]

    # Preprocessing
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean_text = [preprocessor.clean_text(t) for t in train_noisy]
    val_clean_text = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    test_clean_text = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]

    # 3. Fit 20,000 char_wb (2, 5) TF-IDF (matching original candidate)
    print("\n[Step 2] Fitting 20,000-feature TF-IDF (char_wb 2-5, min_df=3)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=True
    )
    X_tr = tfidf.fit_transform(train_clean_text)
    X_va = tfidf.transform(val_clean_text)
    X_te = tfidf.transform(test_clean_text)
    print(f"TF-IDF fitted in {time.time()-t0:.2f}s | Shape: {X_tr.shape}")

    # 4. Train LinearSVC(C=10.0) with CalibratedClassifierCV(method='sigmoid')
    print("\n[Step 3] Training CalibratedClassifierCV(LinearSVC(C=10.0), method='sigmoid')...")
    t0_fit = time.time()
    base_svc = LinearSVC(C=10.0, max_iter=2000, random_state=42)
    cal_svm = CalibratedClassifierCV(base_svc, method="sigmoid")
    cal_svm.fit(X_tr, y_tr)
    fit_time = time.time() - t0_fit
    print(f"Model trained in {fit_time:.2f}s.")

    # 5. Evaluate on Validation and Clean Test Splits
    va_preds = cal_svm.predict(X_va)
    va_acc = accuracy_score(y_va, va_preds)
    te_preds = cal_svm.predict(X_te)
    te_acc = accuracy_score(y_te, te_preds)
    te_f1 = f1_score(y_te, te_preds, average="macro")

    print(f"\nResults for Leakage-Free SVM (20k features, C=10.0, 35% noise):")
    print(f"  Validation Accuracy: {va_acc:.4f} ({va_acc*100:.2f}%)")
    print(f"  Clean Test Accuracy: {te_acc:.4f} ({te_acc*100:.2f}%)")
    print(f"  Clean Test Macro-F1: {te_f1:.4f}")

    # Evaluate on Real Indian Statements (N=105, N=96, N=9, N=58)
    df_own = pd.read_csv(OWN_TEST_PATH)
    df_own["canon"] = df_own["transaction_description"].apply(canonicalize)
    OVERLAPPING_CORES = ["amazon pay", "dominos pizza", "flipkart internet", "myntra designs", "ola cabs", "swiggy", "uber india", "zomato"]
    is_ov_9 = df_own["canon"].apply(lambda c: any(core in c.lower() for core in OVERLAPPING_CORES))
    df_own_96 = df_own[~is_ov_9].copy()
    df_own_9 = df_own[is_ov_9].copy()

    clean_seed_cores = set(df_clean_seed["core"].dropna().str.lower().unique())
    is_in_clean_seed = df_own["canon"].apply(lambda c: any(sc in c.lower() or c.lower() in sc for sc in clean_seed_cores))
    df_own_58 = df_own[~is_in_clean_seed].copy()

    # Predictions
    own_clean_text = [preprocessor.clean_text(t) for t in df_own["transaction_description"]]
    X_own = tfidf.transform(own_clean_text)
    p_own_raw = cal_svm.predict(X_own)
    p_own_cat = [classes[i] for i in p_own_raw]
    p_own_rent = with_rent_rule(df_own["transaction_description"].tolist(), p_own_cat)

    acc_105_raw = accuracy_score(df_own["category"], p_own_cat)
    acc_105_rent = accuracy_score(df_own["category"], p_own_rent)

    # 96 subset
    own_96_clean_text = [preprocessor.clean_text(t) for t in df_own_96["transaction_description"]]
    X_own_96 = tfidf.transform(own_96_clean_text)
    p_96_raw = cal_svm.predict(X_own_96)
    p_96_cat = [classes[i] for i in p_96_raw]
    p_96_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), p_96_cat)
    acc_96_rent = accuracy_score(df_own_96["category"], p_96_rent)

    # 9 subset
    own_9_clean_text = [preprocessor.clean_text(t) for t in df_own_9["transaction_description"]]
    X_own_9 = tfidf.transform(own_9_clean_text)
    p_9_raw = cal_svm.predict(X_own_9)
    p_9_cat = [classes[i] for i in p_9_raw]
    acc_9_raw = accuracy_score(df_own_9["category"], p_9_cat)

    # 58 subset
    own_58_clean_text = [preprocessor.clean_text(t) for t in df_own_58["transaction_description"]]
    X_own_58 = tfidf.transform(own_58_clean_text)
    p_58_raw = cal_svm.predict(X_own_58)
    p_58_cat = [classes[i] for i in p_58_raw]
    p_58_rent = with_rent_rule(df_own_58["transaction_description"].tolist(), p_58_cat)
    acc_58_rent = accuracy_score(df_own_58["category"], p_58_rent)

    print(f"\nReal Indian Statement Accuracies (Matched Config):")
    print(f"  All 105 Raw:             {acc_105_raw*100:.2f}%")
    print(f"  All 105 + Rent Rule:     {acc_105_rent*100:.2f}%")
    print(f"  Non-overlapping 96 +Rent:{acc_96_rent*100:.2f}%")
    print(f"  Overlapping 9 Raw:       {acc_9_raw*100:.2f}%")
    print(f"  Fully Disjoint 58 +Rent: {acc_58_rent*100:.2f}%")

    # 6. Save Artifact
    artifact = {
        "version": "3.1.0-variant_b_leakage_free_20k",
        "model_name": "Variant B Calibrated Linear SVM (char_wb 2-5, 20k feats, Leakage-Free Seed)",
        "vectorizer": tfidf,
        "classes": classes,
        "model": cal_svm,
        "model_type": "tfidf_svm_calibrated",
        "train_samples": len(df_train_clean),
        "metrics": {
            "val_accuracy": round(float(va_acc), 4),
            "test_accuracy": round(float(te_acc), 4),
            "test_macro_f1": round(float(te_f1), 4),
            "indian_105_rent_acc": round(float(acc_105_rent), 4),
            "indian_96_rent_acc": round(float(acc_96_rent), 4),
            "indian_9_raw_acc": round(float(acc_9_raw), 4),
            "indian_58_rent_acc": round(float(acc_58_rent), 4),
            "fit_time_sec": round(float(fit_time), 2)
        }
    }
    joblib.dump(artifact, OUTPUT_MODEL_PATH)
    print(f"\n[SAVED] {OUTPUT_MODEL_PATH.name}")

if __name__ == "__main__":
    main()
