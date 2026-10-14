"""Check Overfitting across Train, Validation, and Test Sets.

Evaluates:
- Candidate Linear SVM
- LightGBM
- XGBoost
- MiniLM+LR

Reports:
- Train Accuracy (N=72,947)
- Validation Accuracy (N=10,075)
- Test Accuracy (N=20,040)
- Overfitting Gaps: (Train - Val) and (Train - Test)
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.predict import TransactionPredictor
from src.preprocessing import TextPreprocessor

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"

def main():
    print("=" * 70)
    print("CHECKING OVERFITTING: TRAIN, VALIDATION, AND TEST ACCURACY")
    print("=" * 70)

    # 1. Load data
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)

    df_train = pd.concat([train_base, df_seed], ignore_index=True)
    print(f"Train samples: {len(df_train):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")

    y_train = df_train["category"].tolist()
    y_val = val_df["category"].tolist()
    y_test = test_df["category"].tolist()

    models = {
        "Candidate_Linear_SVM": MODELS_DIR / "variant_b_candidate.joblib",
        "LightGBM": MODELS_DIR / "lightgbm_transaction_classifier.joblib",
        "XGBoost": MODELS_DIR / "xgboost_transaction_classifier.joblib",
        "MiniLM_LogReg": MODELS_DIR / "minilm_lr_transaction_classifier.joblib",
    }

    results = []

    for name, model_path in models.items():
        print(f"\n--- Evaluating {name} ---")
        if not model_path.exists():
            print(f"Model path {model_path} not found!")
            continue

        predictor = TransactionPredictor(model_path=model_path)

        # Check if MiniLM and cached embeddings exist
        cache_train = REPO_ROOT / "data" / "processed" / "minilm_train.npy"
        cache_val = REPO_ROOT / "data" / "processed" / "minilm_val.npy"
        cache_test = REPO_ROOT / "data" / "processed" / "minilm_test_clean.npy"

        if name == "MiniLM_LogReg" and cache_train.exists() and cache_val.exists() and cache_test.exists():
            print("  Using cached embeddings for MiniLM...")
            t0 = time.time()
            X_tr = np.load(cache_train)
            p_tr_idx = predictor.model.predict(X_tr)
            train_preds = [predictor.classes[i] for i in p_tr_idx]
            train_acc = accuracy_score(y_train, train_preds)
            t_train = time.time() - t0

            t0 = time.time()
            X_va = np.load(cache_val)
            p_va_idx = predictor.model.predict(X_va)
            val_preds = [predictor.classes[i] for i in p_va_idx]
            val_acc = accuracy_score(y_val, val_preds)
            t_val = time.time() - t0

            t0 = time.time()
            X_te = np.load(cache_test)
            p_te_idx = predictor.model.predict(X_te)
            test_preds = [predictor.classes[i] for i in p_te_idx]
            test_acc = accuracy_score(y_test, test_preds)
            t_test = time.time() - t0
        else:
            # Predict Train
            t0 = time.time()
            res_train = predictor.predict_batch(df_train[["transaction_description"]])
            train_preds = res_train["predicted_category"].tolist()
            train_acc = accuracy_score(y_train, train_preds)
            t_train = time.time() - t0

            # Predict Val
            t0 = time.time()
            res_val = predictor.predict_batch(val_df[["transaction_description"]])
            val_preds = res_val["predicted_category"].tolist()
            val_acc = accuracy_score(y_val, val_preds)
            t_val = time.time() - t0

            # Predict Test
            t0 = time.time()
            res_test = predictor.predict_batch(test_df[["transaction_description"]])
            test_preds = res_test["predicted_category"].tolist()
            test_acc = accuracy_score(y_test, test_preds)
            t_test = time.time() - t0

        gap_train_val = train_acc - val_acc
        gap_train_test = train_acc - test_acc

        print(f"  Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%) [took {t_train:.1f}s]")
        print(f"  Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%) [took {t_val:.1f}s]")
        print(f"  Test Acc:  {test_acc:.4f} ({test_acc*100:.2f}%) [took {t_test:.1f}s]")
        print(f"  Gap (Train - Val):  {gap_train_val*100:+.2f}%")
        print(f"  Gap (Train - Test): {gap_train_test*100:+.2f}%")

        results.append({
            "Model": name,
            "Train_Acc": train_acc,
            "Val_Acc": val_acc,
            "Test_Acc": test_acc,
            "Gap_Train_Val": gap_train_val,
            "Gap_Train_Test": gap_train_test
        })

    df_res = pd.DataFrame(results)
    out_path = REPO_ROOT / "results" / "overfitting_check.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n[SAVED] Overfitting results saved to {out_path}")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
