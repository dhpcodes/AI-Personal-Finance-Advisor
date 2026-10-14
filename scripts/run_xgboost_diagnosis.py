"""Diagnostic Script for XGBoost Performance on Sparse Text (Task 4).

Checks:
1. Class order and label encoding comparison between LightGBM and XGBoost.
2. Feature vectorizer vocabulary, sparsity, and row alignment.
3. Run XGBoost with max_depth=8, learning_rate=0.1 on full data.
4. Run XGBoost with tree_method='exact' on a 20,000-row subsample.
5. Outputs validation and test accuracy for all configurations.
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
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import TextPreprocessor

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"

def main():
    print("=" * 70)
    print("TASK 4: XGBOOST EXPERIMENTAL DIAGNOSIS ON SPARSE TEXT")
    print("=" * 70)

    # 1. Inspect existing LightGBM and XGBoost artifacts
    print("\n[Step 1] Comparing Existing LightGBM vs XGBoost Artifacts...")
    lgb_art = joblib.load(MODELS_DIR / "lightgbm_transaction_classifier.joblib")
    xgb_art = joblib.load(MODELS_DIR / "xgboost_transaction_classifier.joblib")

    print(f"  LightGBM Classes ({len(lgb_art['classes'])}): {lgb_art['classes']}")
    print(f"  XGBoost  Classes ({len(xgb_art['classes'])}): {xgb_art['classes']}")
    classes_identical = lgb_art['classes'] == xgb_art['classes']
    print(f"  >> Classes 100% Identical: {classes_identical}")

    lgb_vocab = lgb_art['vectorizer'].vocabulary_
    xgb_vocab = xgb_art['vectorizer'].vocabulary_
    vocab_identical = lgb_vocab == xgb_vocab
    print(f"  >> Vocabulary 100% Identical: {vocab_identical} ({len(lgb_vocab)} features)")

    # 2. Load Data Splits
    print("\n[Step 2] Loading Data Splits...")
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)

    df_train = pd.concat([train_base, df_seed], ignore_index=True)
    print(f"  Train: {len(df_train):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in df_train["transaction_description"]]
    val_clean = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    test_clean = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]

    le = LabelEncoder().fit(df_train["category"])
    classes = list(le.classes_)
    y_tr = le.transform(df_train["category"])
    y_va = le.transform(val_df["category"])
    y_te = le.transform(test_df["category"])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=5,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    X_tr = vectorizer.fit_transform(train_clean)
    X_va = vectorizer.transform(val_clean)
    X_te = vectorizer.transform(test_clean)

    results = []

    # 3. Run XGBoost with max_depth=8, learning_rate=0.1
    print("\n[Step 3] Training XGBoost with max_depth=8, learning_rate=0.1 (full data)...")
    xgb_depth8 = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=100,
        learning_rate=0.10,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss"
    )
    t0 = time.time()
    xgb_depth8.fit(X_tr, y_tr)
    dur_d8 = time.time() - t0

    va_preds_d8 = xgb_depth8.predict(X_va)
    va_acc_d8 = accuracy_score(y_va, va_preds_d8)
    te_preds_d8 = xgb_depth8.predict(X_te)
    te_acc_d8 = accuracy_score(y_te, te_preds_d8)

    print(f"  XGBoost (depth=8, lr=0.1): Fit={dur_d8:.1f}s | Val Acc={va_acc_d8*100:.2f}% | Test Acc={te_acc_d8*100:.2f}%")
    results.append({
        "Configuration": "XGBoost (max_depth=8, lr=0.10, full 72k)",
        "Val_Accuracy": va_acc_d8,
        "Test_Accuracy": te_acc_d8,
        "Fit_Time_Sec": dur_d8
    })

    joblib.dump({
        "version": "1.0.0-xgboost_depth8",
        "model_name": "XGBoost (depth=8, lr=0.1)",
        "vectorizer": vectorizer,
        "classes": classes,
        "model": xgb_depth8,
        "model_type": "tfidf_xgboost"
    }, MODELS_DIR / "xgboost_depth8_lr01.joblib")

    # 4. Run XGBoost with tree_method='exact' on a 20,000-row subsample
    print("\n[Step 4] Training XGBoost with tree_method='exact' on 20,000-row subsample...")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=20000, random_state=42)
    sub_idx, _ = next(sss.split(df_train, df_train["category"]))
    X_tr_20k = X_tr[sub_idx]
    y_tr_20k = y_tr[sub_idx]

    # Exact tree_method evaluates all possible split points without histogram binning
    xgb_exact = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=50,
        learning_rate=0.20,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="exact",
        eval_metric="mlogloss"
    )
    t0 = time.time()
    xgb_exact.fit(X_tr_20k, y_tr_20k)
    dur_exact = time.time() - t0

    va_preds_ex = xgb_exact.predict(X_va)
    va_acc_ex = accuracy_score(y_va, va_preds_ex)
    te_preds_ex = xgb_exact.predict(X_te)
    te_acc_ex = accuracy_score(y_te, te_preds_ex)

    print(f"  XGBoost (tree_method='exact', 20k rows): Fit={dur_exact:.1f}s | Val Acc={va_acc_ex*100:.2f}% | Test Acc={te_acc_ex*100:.2f}%")
    results.append({
        "Configuration": "XGBoost (tree_method='exact', 20k rows)",
        "Val_Accuracy": va_acc_ex,
        "Test_Accuracy": te_acc_ex,
        "Fit_Time_Sec": dur_exact
    })

    joblib.dump({
        "version": "1.0.0-xgboost_exact_20k",
        "model_name": "XGBoost Exact (20k Subsample)",
        "vectorizer": vectorizer,
        "classes": classes,
        "model": xgb_exact,
        "model_type": "tfidf_xgboost"
    }, MODELS_DIR / "xgboost_exact_20k.joblib")

    # Summary
    df_diag = pd.DataFrame(results)
    out_path = RESULTS_DIR / "xgboost_diagnosis_results.csv"
    df_diag.to_csv(out_path, index=False)
    print(f"\n[SAVED] XGBoost diagnosis saved to {out_path}")
    print(df_diag.to_string(index=False))

if __name__ == "__main__":
    main()
