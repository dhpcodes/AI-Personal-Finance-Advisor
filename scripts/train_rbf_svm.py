"""Train Non-Linear RBF Kernel SVM vs Linear SVM on Equal 20k Subsample (Task 2).

Specifications:
- Stratified 20,000-row subsample of training set (fixed seed=42).
- Trains RBF-kernel SVC and LinearSVC on the EXACT same 20,000 rows.
- Tunes C (and gamma for RBF) on the validation set only.
- Evaluates on clean test, reporting train, val, and test accuracy.
- Saves:
  - models/rbf_svm_20k.joblib
  - models/linear_svm_20k.joblib
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import TextPreprocessor

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"

RBF_MODEL_PATH = MODELS_DIR / "rbf_svm_20k.joblib"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_svm_20k.joblib"

def main():
    print("=" * 70)
    print("TASK 2: RBF-KERNEL SVM VS LINEAR SVM ON EQUAL 20K SUBSAMPLE")
    print("=" * 70)

    # 1. Load data
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)

    df_train_full = pd.concat([train_base, df_seed], ignore_index=True)
    print(f"Full Train Set: {len(df_train_full):,} rows")

    # 2. Stratified 20,000-row Subsample
    print("\n[Step 1] Creating stratified 20,000-row training subsample (seed=42)...")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=20000, random_state=42)
    sub_idx, _ = next(sss.split(df_train_full, df_train_full["category"]))
    df_train_20k = df_train_full.iloc[sub_idx].copy().reset_index(drop=True)
    print(f"Subsampled Train: {len(df_train_20k):,} rows")

    # 3. Canonicalized Preprocessing
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in df_train_20k["transaction_description"]]
    val_clean = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    test_clean = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]

    le = LabelEncoder().fit(df_train_full["category"])
    classes = list(le.classes_)
    y_tr_20k = le.transform(df_train_20k["category"])
    y_va = le.transform(val_df["category"])
    y_te = le.transform(test_df["category"])

    # 4. TF-IDF Feature Extraction (3,000 features, char_wb 2-4)
    print("\n[Step 2] Fitting TF-IDF Vectorizer on 20,000-row subsample...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    t0 = time.time()
    X_tr = vectorizer.fit_transform(train_clean)
    X_va = vectorizer.transform(val_clean)
    X_te = vectorizer.transform(test_clean)
    print(f"TF-IDF fitted in {time.time()-t0:.2f}s | Shape: {X_tr.shape}")

    # 5. Tune & Train Linear SVM on 20k
    if LINEAR_MODEL_PATH.exists():
        print(f"  [CACHE HIT] {LINEAR_MODEL_PATH.name} already exists. Skipping linear SVM training.")
    else:
        print("\n[Step 3] Tuning & Training Linear SVM on 20k rows...")
        c_candidates = [0.1, 1.0, 5.0, 10.0]
        best_c_lin = None
        best_lin_val_acc = -1.0
        best_lin_model = None
        best_lin_fit_time = 0.0

        for c_val in c_candidates:
            lin_svc = LinearSVC(C=c_val, random_state=42, max_iter=2000)
            t_start = time.time()
            lin_svc.fit(X_tr, y_tr_20k)
            dur = time.time() - t_start
            v_preds = lin_svc.predict(X_va)
            v_acc = accuracy_score(y_va, v_preds)
            print(f"  LinearSVC(C={c_val}): Fit={dur:.2f}s | Val Acc={v_acc*100:.2f}%")
            if v_acc > best_lin_val_acc:
                best_lin_val_acc = v_acc
                best_c_lin = c_val
                best_lin_model = lin_svc
                best_lin_fit_time = dur

        # Calibrate best Linear SVM
        print(f"  Best Linear SVM C={best_c_lin} (Val Acc: {best_lin_val_acc*100:.2f}%). Calibrating...")
        cal_lin = CalibratedClassifierCV(best_lin_model, cv=3)
        t0_cal = time.time()
        cal_lin.fit(X_tr, y_tr_20k)
        best_lin_fit_time += (time.time() - t0_cal)

        lin_test_preds = cal_lin.predict(X_te)
        lin_test_acc = accuracy_score(y_te, lin_test_preds)
        lin_train_preds = cal_lin.predict(X_tr)
        lin_train_acc = accuracy_score(y_tr_20k, lin_train_preds)
        print(f"  Linear SVM (20k) -> Train: {lin_train_acc*100:.2f}% | Val: {best_lin_val_acc*100:.2f}% | Clean Test: {lin_test_acc*100:.2f}%")

        joblib.dump({
            "version": "1.0.0-linear_svm_20k",
            "model_name": "Linear SVM (20k Subsample, char_wb 2-4)",
            "vectorizer": vectorizer,
            "classes": classes,
            "model": cal_lin,
            "model_type": "tfidf_svm_calibrated",
            "best_C": best_c_lin,
            "train_samples": 20000,
            "metrics": {
                "train_accuracy": round(float(lin_train_acc), 4),
                "val_accuracy": round(float(best_lin_val_acc), 4),
                "test_accuracy": round(float(lin_test_acc), 4),
                "fit_time_sec": round(float(best_lin_fit_time), 2)
            }
        }, LINEAR_MODEL_PATH)
        print(f"  [SAVED] {LINEAR_MODEL_PATH.name}")

    # 6. Tune & Train RBF-Kernel SVM on 20k
    print("\n[Step 4] Tuning & Training RBF-Kernel SVM on 20k rows...")
    rbf_configs = [
        {"C": 1.0, "gamma": "scale"},
        {"C": 5.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 5.0, "gamma": 0.1}
    ]
    best_rbf_cfg = None
    best_rbf_val_acc = -1.0
    best_rbf_model = None
    best_rbf_fit_time = 0.0

    for cfg in rbf_configs:
        rbf_svc = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", probability=False, random_state=42)
        t_start = time.time()
        rbf_svc.fit(X_tr, y_tr_20k)
        dur = time.time() - t_start
        v_preds = rbf_svc.predict(X_va)
        v_acc = accuracy_score(y_va, v_preds)
        print(f"  SVC(RBF, C={cfg['C']}, gamma={cfg['gamma']}): Fit={dur:.1f}s | Val Acc={v_acc*100:.2f}%")
        if v_acc > best_rbf_val_acc:
            best_rbf_val_acc = v_acc
            best_rbf_cfg = cfg
            best_rbf_model = rbf_svc
            best_rbf_fit_time = dur

    rbf_test_preds = best_rbf_model.predict(X_te)
    rbf_test_acc = accuracy_score(y_te, rbf_test_preds)
    rbf_train_preds = best_rbf_model.predict(X_tr)
    rbf_train_acc = accuracy_score(y_tr_20k, rbf_train_preds)
    print(f"\nRBF SVM (20k) -> Train: {rbf_train_acc*100:.2f}% | Val: {best_rbf_val_acc*100:.2f}% | Clean Test: {rbf_test_acc*100:.2f}%")

    joblib.dump({
        "version": "1.0.0-rbf_svm_20k",
        "model_name": "RBF Kernel SVM (20k Subsample, char_wb 2-4)",
        "vectorizer": vectorizer,
        "classes": classes,
        "model": best_rbf_model,
        "model_type": "tfidf_svm_calibrated",
        "best_config": best_rbf_cfg,
        "train_samples": 20000,
        "metrics": {
            "train_accuracy": round(float(rbf_train_acc), 4),
            "val_accuracy": round(float(best_rbf_val_acc), 4),
            "test_accuracy": round(float(rbf_test_acc), 4),
            "fit_time_sec": round(float(best_rbf_fit_time), 2)
        }
    }, RBF_MODEL_PATH)
    print(f"  [SAVED] {RBF_MODEL_PATH.name}")

if __name__ == "__main__":
    main()
