"""Deep Diagnosis of XGBoost on Sparse Text (Task 2).

Tests:
(a) XGBoost with grow_policy='lossguide', max_leaves=63, same trees (100) and lr (0.20) as LightGBM on full data.
(b) The same run with the TF-IDF matrix passed as dense (or explicit zeros) to test the missing-vs-zero hypothesis.
    (Tested on both full 72k and 20k subsample).
(c) LightGBM on the same 20k subsample for a fair reference against XGBoost's 20k benchmarks.

Outputs:
- Train, val, and test accuracy for each configuration.
- Saves results to results/xgboost_deep_diagnosis.csv.
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from src.preprocessing import TextPreprocessor

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"

def apply_single_noise(text: str, rng, kind: str) -> str:
    import re
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
    import random
    print("=" * 75)
    print("TASK 2: XGBOOST DEEP DIAGNOSIS (LOSSGUIDE, DENSE ZEROS, 20K CONTROL)")
    print("=" * 75)

    # 1. Load Data
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)

    train_df = pd.concat([train_base, df_seed], ignore_index=True)
    print(f"Full Training Set: {len(train_df):,} rows")

    # 2. 35% Noise Augmentation (seed 42)
    print("\n[Step 1] Applying 35% noise augmentation (seed 42)...")
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train_df)) < 0.35
    train_noisy = [
        apply_single_noise(t, r_tr, r_tr.choice(["upi", "pos", "upper", "lower", "trunc", "typo"])) if is_n else t
        for is_n, t in zip(m_tr, train_df["transaction_description"])
    ]

    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in train_noisy]
    val_clean = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    test_clean = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]

    le = LabelEncoder().fit(train_df["category"])
    classes = list(le.classes_)
    y_tr = le.transform(train_df["category"])
    y_va = le.transform(val_df["category"])
    y_te = le.transform(test_df["category"])

    # 3. TF-IDF 3,000 Features (char_wb 2-4)
    print("\n[Step 2] Fitting 3,000 TF-IDF features...")
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=5,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    t0 = time.time()
    X_tr_sparse = vec.fit_transform(train_clean).astype(np.float32)
    X_va_sparse = vec.transform(val_clean).astype(np.float32)
    X_te_sparse = vec.transform(test_clean).astype(np.float32)
    print(f"  TF-IDF fitted in {time.time()-t0:.2f}s | Shape: {X_tr_sparse.shape}")

    records = []

    # -------------------------------------------------------------
    # (a) XGBoost: grow_policy='lossguide', max_leaves=63, trees=100, lr=0.20
    # -------------------------------------------------------------
    print("\n[Step 3] (a) XGBoost with grow_policy='lossguide', max_leaves=63 (Sparse)...")
    xgb_lossguide = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=100,
        learning_rate=0.20,
        max_leaves=63,
        grow_policy="lossguide",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss"
    )
    t0 = time.time()
    xgb_lossguide.fit(X_tr_sparse, y_tr)
    dur_a = time.time() - t0

    tr_acc_a = accuracy_score(y_tr, xgb_lossguide.predict(X_tr_sparse))
    va_acc_a = accuracy_score(y_va, xgb_lossguide.predict(X_va_sparse))
    te_acc_a = accuracy_score(y_te, xgb_lossguide.predict(X_te_sparse))
    print(f"  XGBoost lossguide (Sparse) -> Train: {tr_acc_a*100:.2f}% | Val: {va_acc_a*100:.2f}% | Test: {te_acc_a*100:.2f}% (Fit: {dur_a:.1f}s)")
    records.append({
        "Experiment": "(a) XGBoost lossguide (max_leaves=63, lr=0.20, sparse)",
        "Data_Scale": f"Full ({len(train_df):,} rows)",
        "Matrix_Type": "Sparse CSR",
        "Train_Accuracy": round(tr_acc_a, 4),
        "Val_Accuracy": round(va_acc_a, 4),
        "Test_Accuracy": round(te_acc_a, 4),
        "Fit_Time_Sec": round(dur_a, 2)
    })

    # -------------------------------------------------------------
    # (b) XGBoost: Same lossguide run with dense matrix / explicit zeros
    # -------------------------------------------------------------
    print("\n[Step 4] (b) XGBoost with dense TF-IDF matrix (testing missing-vs-zero hypothesis)...")
    print("  Converting matrices to dense float32 arrays...")
    t0_conv = time.time()
    X_tr_dense = X_tr_sparse.toarray()
    X_va_dense = X_va_sparse.toarray()
    X_te_dense = X_te_sparse.toarray()
    print(f"  Converted to dense in {time.time()-t0_conv:.2f}s | Train memory: {X_tr_dense.nbytes / 1024**2:.1f} MB")

    xgb_dense = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=100,
        learning_rate=0.20,
        max_leaves=63,
        grow_policy="lossguide",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss"
    )
    t0 = time.time()
    xgb_dense.fit(X_tr_dense, y_tr)
    dur_b = time.time() - t0

    tr_acc_b = accuracy_score(y_tr, xgb_dense.predict(X_tr_dense))
    va_acc_b = accuracy_score(y_va, xgb_dense.predict(X_va_dense))
    te_acc_b = accuracy_score(y_te, xgb_dense.predict(X_te_dense))
    print(f"  XGBoost lossguide (Dense)  -> Train: {tr_acc_b*100:.2f}% | Val: {va_acc_b*100:.2f}% | Test: {te_acc_b*100:.2f}% (Fit: {dur_b:.1f}s)")
    records.append({
        "Experiment": "(b) XGBoost lossguide (max_leaves=63, lr=0.20, dense zeros)",
        "Data_Scale": f"Full ({len(train_df):,} rows)",
        "Matrix_Type": "Dense ndarray",
        "Train_Accuracy": round(tr_acc_b, 4),
        "Val_Accuracy": round(va_acc_b, 4),
        "Test_Accuracy": round(te_acc_b, 4),
        "Fit_Time_Sec": round(dur_b, 2)
    })

    # Clean up dense memory
    del X_tr_dense, X_va_dense, X_te_dense

    # -------------------------------------------------------------
    # (c) LightGBM on the exact same 20k subsample (Fair Reference)
    # -------------------------------------------------------------
    print("\n[Step 5] (c) LightGBM on Stratified 20k Subsample (Fair Control)...")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=20000, random_state=42)
    sub_idx, _ = next(sss.split(train_df, train_df["category"]))
    X_tr_20k = X_tr_sparse[sub_idx]
    y_tr_20k = y_tr[sub_idx]

    lgb_20k = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.20,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    t0 = time.time()
    lgb_20k.fit(X_tr_20k, y_tr_20k)
    dur_c = time.time() - t0

    tr_acc_c = accuracy_score(y_tr_20k, lgb_20k.predict(X_tr_20k))
    va_acc_c = accuracy_score(y_va, lgb_20k.predict(X_va_sparse))
    te_acc_c = accuracy_score(y_te, lgb_20k.predict(X_te_sparse))
    print(f"  LightGBM (20k Subsample)   -> Train: {tr_acc_c*100:.2f}% | Val: {va_acc_c*100:.2f}% | Test: {te_acc_c*100:.2f}% (Fit: {dur_c:.1f}s)")
    records.append({
        "Experiment": "(c) LightGBM (num_leaves=63, lr=0.20, 20k control)",
        "Data_Scale": "Subsample (20,000 rows)",
        "Matrix_Type": "Sparse CSR",
        "Train_Accuracy": round(tr_acc_c, 4),
        "Val_Accuracy": round(va_acc_c, 4),
        "Test_Accuracy": round(te_acc_c, 4),
        "Fit_Time_Sec": round(dur_c, 2)
    })

    # Summary table
    df_diag = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "xgboost_deep_diagnosis.csv"
    df_diag.to_csv(out_csv, index=False)
    print(f"\n[SAVED] XGBoost deep diagnosis saved to {out_csv}")
    print(df_diag.to_string(index=False))

if __name__ == "__main__":
    main()
