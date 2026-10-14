"""Leakage-Removed Indian Experiment & XGBoost Early-Stopping Verification.

Tasks:
1. Remove 8 overlapping merchant cores from data/indian_merchants_train_seed.csv:
   ('amazon pay', 'dominos pizza', 'flipkart internet', 'myntra designs',
    'ola cabs', 'swiggy', 'uber india', 'zomato')
2. Retrain:
   - Linear SVM -> models/variant_b_candidate_leakage_free.joblib
   - LightGBM   -> models/lightgbm_leakage_free.joblib
   - XGBoost    -> models/xgboost_leakage_free.joblib
   - MiniLM+LR  -> models/minilm_lr_leakage_free.joblib
3. Rerun XGBoost with early stopping on validation set & tree budget=100 (same as LightGBM).
4. Evaluate before vs. after on:
   - All 105 real Indian statements (N=105)
   - Non-overlapping 96 statements (N=96)
   - The 9 overlapping statements (N=9)
   - Statements with zero merchant match in reduced seed (N=58)
   - Clean test set (N=20,040)
5. Generate comparison tables and McNemar tests.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from src.preprocessing import TextPreprocessor
from src.predict import TransactionPredictor
from src.model_comparison_suite import with_rent_rule

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"

OVERLAPPING_CORES = [
    "amazon pay",
    "dominos pizza",
    "flipkart internet",
    "myntra designs",
    "ola cabs",
    "swiggy",
    "uber india",
    "zomato"
]

def wilson_score_interval(acc: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    z = 1.95996
    denom = 1 + z**2 / n
    centre_adj = acc + z**2 / (2 * n)
    adj_val = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre_adj - adj_val) / denom)
    upper = min(1.0, (centre_adj + adj_val) / denom)
    return lower, upper

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    a_correct = np.array([ya == yt for ya, yt in zip(y_pred_a, y_true)])
    b_correct = np.array([yb == yt for yb, yt in zip(y_pred_b, y_true)])
    n10 = int(np.sum(a_correct & (~b_correct)))
    n01 = int(np.sum((~a_correct) & b_correct))
    if n10 + n01 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n10": 0, "n01": 0}
    stat = ((abs(n10 - n01) - 1.0) ** 2) / (n10 + n01)
    p_val = float(chi2.sf(stat, df=1))
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_val), 5), "n10": n10, "n01": n01}

def main():
    print("=" * 70)
    print("LEAKAGE-REMOVED INDIAN EXPERIMENT & XGBOOST AUDIT")
    print("=" * 70)

    # 1. Prepare Reduced Seed
    print("\n[Step 1] Preparing leakage-free training seed...")
    df_seed = pd.read_csv(SEED_PATH)
    n_orig_seed = len(df_seed)
    
    # Filter out overlapping cores
    mask_overlap = df_seed["core"].isin(OVERLAPPING_CORES)
    df_seed_clean = df_seed[~mask_overlap].copy()
    n_clean_seed = len(df_seed_clean)
    n_removed = mask_overlap.sum()
    
    clean_seed_path = DATA_DIR / "indian_merchants_train_seed_leakage_free.csv"
    df_seed_clean.to_csv(clean_seed_path, index=False)
    print(f"Original Indian seed: {n_orig_seed:,} rows")
    print(f"Removed overlapping rows: {n_removed:,} rows ({n_removed/n_orig_seed*100:.2f}%)")
    print(f"Reduced Indian seed: {n_clean_seed:,} rows saved to {clean_seed_path.name}")

    # 2. Combine with Train Split
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()

    df_train_clean = pd.concat([train_base, df_seed_clean], ignore_index=True)
    print(f"Total Reduced Training Set: {len(df_train_clean):,} rows (69,885 Mitul + {n_clean_seed} seed)")

    # 3. Partition Real Indian Statements
    df_own = pd.read_csv(OWN_TEST_PATH)
    df_own["canon"] = df_own["transaction_description"].apply(canonicalize)
    
    # Subset 1: All 105 rows
    # Subset 2: 96 rows (excluding the 8 overlapping cores)
    is_ov_9 = df_own["canon"].apply(lambda c: any(core in c.lower() for core in OVERLAPPING_CORES))
    df_own_96 = df_own[~is_ov_9].copy()
    df_own_9 = df_own[is_ov_9].copy()

    # Subset 3: 58 rows (no match against ANY core in the reduced seed)
    clean_seed_cores = set(df_seed_clean["core"].dropna().str.lower().unique())
    is_in_clean_seed = df_own["canon"].apply(lambda c: any(sc in c.lower() or c.lower() in sc for sc in clean_seed_cores))
    df_own_58 = df_own[~is_in_clean_seed].copy()

    print(f"\nIndian Statement Evaluation Partitions:")
    print(f"  All Indian Statements:          N = {len(df_own)}")
    print(f"  Non-overlapping Subset:         N = {len(df_own_96)} (excludes 8 removed cores)")
    print(f"  Overlapping Brand Subset:       N = {len(df_own_9)} (the 8 removed cores)")
    print(f"  Fully Disjoint Merchant Subset: N = {len(df_own_58)} (zero match in reduced seed)")

    preprocessor = TextPreprocessor(use_canonicalization=True)
    X_train_clean_text = [preprocessor.clean_text(t) for t in df_train_clean["transaction_description"]]
    y_train_clean = df_train_clean["category"].tolist()

    X_val_text = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    y_val = val_df["category"].tolist()

    X_test_text = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]
    y_test = test_df["category"].tolist()

    # Label Encoder
    classes = sorted(list(set(y_train_clean)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_clean_idx = np.array([class_to_idx[c] for c in y_train_clean])
    y_val_idx = np.array([class_to_idx[c] for c in y_val])
    y_test_idx = np.array([class_to_idx[c] for c in y_test])

    # TF-IDF Feature Extractor (strictly 3,000 subwords)
    print("\n[Step 2] Fitting TF-IDF Vectorizer on reduced training set...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=5,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    X_train_vec = vectorizer.fit_transform(X_train_clean_text)
    X_val_vec = vectorizer.transform(X_val_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 4. Retrain Models
    print("\n[Step 3] Retraining Models on Reduced Training Set...")

    # 4a. Linear SVM
    svm_clean_path = MODELS_DIR / "variant_b_candidate_leakage_free.joblib"
    if svm_clean_path.exists():
        print(f"  [CACHE HIT] {svm_clean_path.name} already exists. Skipping training.")
    else:
        print("  -> Training Linear SVM (Calibrated)...")
        svm_base = LinearSVC(C=1.0, random_state=42)
        cal_svm = CalibratedClassifierCV(estimator=svm_base, cv=3)
        t0 = time.time()
        cal_svm.fit(X_train_vec, y_train_clean_idx)
        dur_svm = time.time() - t0
        joblib.dump({
            "model": cal_svm,
            "vectorizer": vectorizer,
            "classes": classes,
            "preprocessor": preprocessor,
            "model_type": "tfidf_svm_calibrated",
            "model_name": "Candidate_Linear_SVM_Leakage_Free",
            "train_samples": len(df_train_clean),
            "fit_time_sec": round(dur_svm, 2)
        }, svm_clean_path)
        print(f"     Saved {svm_clean_path.name} in {dur_svm:.1f}s.")

    # 4b. LightGBM
    lgbm_clean_path = MODELS_DIR / "lightgbm_leakage_free.joblib"
    if lgbm_clean_path.exists():
        print(f"  [CACHE HIT] {lgbm_clean_path.name} already exists. Skipping training.")
    else:
        print("  -> Training LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.15,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        t0 = time.time()
        lgbm.fit(X_train_vec, y_train_clean_idx)
        dur_lgbm = time.time() - t0
        joblib.dump({
            "model": lgbm,
            "vectorizer": vectorizer,
            "classes": classes,
            "preprocessor": preprocessor,
            "model_type": "tfidf_lightgbm",
            "model_name": "LightGBM_Leakage_Free",
            "train_samples": len(df_train_clean),
            "fit_time_sec": round(dur_lgbm, 2)
        }, lgbm_clean_path)
        print(f"     Saved {lgbm_clean_path.name} in {dur_lgbm:.1f}s.")

    # 4c. XGBoost Standard (n_estimators=80, lr=0.20)
    xgb_clean_path = MODELS_DIR / "xgboost_leakage_free.joblib"
    if xgb_clean_path.exists():
        print(f"  [CACHE HIT] {xgb_clean_path.name} already exists. Skipping training.")
    else:
        print("  -> Training XGBoost (standard budget)...")
        xgb = XGBClassifier(
            n_estimators=80,
            learning_rate=0.20,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            n_jobs=-1
        )
        t0 = time.time()
        xgb.fit(X_train_vec, y_train_clean_idx)
        dur_xgb = time.time() - t0
        joblib.dump({
            "model": xgb,
            "vectorizer": vectorizer,
            "classes": classes,
            "preprocessor": preprocessor,
            "model_type": "tfidf_xgboost",
            "model_name": "XGBoost_Leakage_Free",
            "train_samples": len(df_train_clean),
            "fit_time_sec": round(dur_xgb, 2)
        }, xgb_clean_path)
        print(f"     Saved {xgb_clean_path.name} in {dur_xgb:.1f}s.")

    # 4d. Task 4: XGBoost with Early Stopping on Validation Set & Tree Budget=100 (same as LightGBM)
    xgb_es_path = MODELS_DIR / "xgboost_early_stopping_100trees.joblib"
    if xgb_es_path.exists():
        print(f"  [CACHE HIT] {xgb_es_path.name} already exists. Skipping training.")
    else:
        print("\n[Step 4] Running Task 4: XGBoost with Early Stopping (100 trees budget)...")
        xgb_es = XGBClassifier(
            n_estimators=100,
            learning_rate=0.15,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=10,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1
        )
        t0 = time.time()
        xgb_es.fit(X_train_vec, y_train_clean_idx, eval_set=[(X_val_vec, y_val_idx)], verbose=False)
        dur_xgb_es = time.time() - t0
        best_iter = getattr(xgb_es, "best_iteration", 100)
        joblib.dump({
            "model": xgb_es,
            "vectorizer": vectorizer,
            "classes": classes,
            "preprocessor": preprocessor,
            "model_type": "tfidf_xgboost",
            "model_name": "XGBoost_EarlyStopping_100Trees",
            "best_iteration": int(best_iter),
            "fit_time_sec": round(dur_xgb_es, 2)
        }, xgb_es_path)
        print(f"     XGBoost Early Stopping reached best iteration: {best_iter} in {dur_xgb_es:.1f}s.")

    # 4e. MiniLM + Logistic Regression on Reduced Seed
    minilm_clean_path = MODELS_DIR / "minilm_lr_leakage_free.joblib"
    if minilm_clean_path.exists():
        print(f"  [CACHE HIT] {minilm_clean_path.name} already exists. Skipping training.")
    else:
        print("\n  -> Encoding Reduced Training Set with MiniLM-L6-v2...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        t0 = time.time()
        embs_train_clean_path = DATA_DIR / "processed" / "minilm_train_leakage_free.npy"
        if embs_train_clean_path.exists():
            print(f"     [CACHE HIT] Loaded {embs_train_clean_path.name}")
            X_train_emb = np.load(embs_train_clean_path)
        else:
            print(f"     Encoding {len(X_train_clean_text):,} texts...")
            X_train_emb = encoder.encode(X_train_clean_text, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
            np.save(embs_train_clean_path, X_train_emb)
        dur_emb = time.time() - t0

        print("  -> Training Logistic Regression on MiniLM embeddings...")
        clf_minilm = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        t0 = time.time()
        clf_minilm.fit(X_train_emb, y_train_clean_idx)
        dur_lr = time.time() - t0

        joblib.dump({
            "model": clf_minilm,
            "encoder_name": "all-MiniLM-L6-v2",
            "classes": classes,
            "preprocessor": preprocessor,
            "model_type": "minilm_embeddings",
            "model_name": "MiniLM_LogReg_Leakage_Free",
            "train_samples": len(df_train_clean),
            "fit_time_sec": round(dur_lr, 2)
        }, minilm_clean_path)
        print(f"     Saved {minilm_clean_path.name} in {dur_lr:.1f}s.")

    # 5. Evaluate Before vs After Leakage Removal
    print("\n[Step 5] Evaluating Models Before vs After Leakage Removal...")

    models_to_eval = {
        # Original models
        "SVM (Original Seed)": MODELS_DIR / "variant_b_candidate.joblib",
        "LightGBM (Original Seed)": MODELS_DIR / "lightgbm_transaction_classifier.joblib",
        "XGBoost (Original Seed)": MODELS_DIR / "xgboost_transaction_classifier.joblib",
        "MiniLM+LR (Original Seed)": MODELS_DIR / "minilm_lr_transaction_classifier.joblib",
        
        # Leakage-free models
        "SVM (Leakage-Free)": svm_clean_path,
        "LightGBM (Leakage-Free)": lgbm_clean_path,
        "XGBoost (Leakage-Free)": xgb_clean_path,
        "MiniLM+LR (Leakage-Free)": minilm_clean_path,
        
        # Task 4 Model
        "XGBoost (ES 100 Trees)": xgb_es_path
    }

    eval_rows = []
    preds_dict = {}

    for m_label, m_path in models_to_eval.items():
        pred_obj = TransactionPredictor(model_path=m_path)
        
        # (a) All 105 rows
        r105 = pred_obj.predict_batch(df_own[["transaction_description"]])
        p105 = r105["predicted_category"].tolist()
        p105_rent = with_rent_rule(df_own["transaction_description"].tolist(), p105)
        acc_105_raw = accuracy_score(df_own["category"], p105)
        acc_105_rent = accuracy_score(df_own["category"], p105_rent)

        # (b) Non-overlapping 96 rows
        r96 = pred_obj.predict_batch(df_own_96[["transaction_description"]])
        p96 = r96["predicted_category"].tolist()
        p96_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), p96)
        acc_96_raw = accuracy_score(df_own_96["category"], p96)
        acc_96_rent = accuracy_score(df_own_96["category"], p96_rent)

        # (c) Overlapping 9 rows
        r9 = pred_obj.predict_batch(df_own_9[["transaction_description"]])
        p9 = r9["predicted_category"].tolist()
        acc_9_raw = accuracy_score(df_own_9["category"], p9)

        # (d) Fully disjoint 58 rows
        r58 = pred_obj.predict_batch(df_own_58[["transaction_description"]])
        p58 = r58["predicted_category"].tolist()
        p58_rent = with_rent_rule(df_own_58["transaction_description"].tolist(), p58)
        acc_58_raw = accuracy_score(df_own_58["category"], p58)
        acc_58_rent = accuracy_score(df_own_58["category"], p58_rent)

        # Clean test set (N=20,040)
        r_clean = pred_obj.predict_batch(test_df[["transaction_description"]])
        p_clean = r_clean["predicted_category"].tolist()
        acc_clean = accuracy_score(y_test, p_clean)

        preds_dict[m_label] = {
            "p105": p105_rent,
            "p96": p96_rent,
            "p_clean": p_clean
        }

        eval_rows.append({
            "Model_Configuration": m_label,
            "Clean_Test_Acc_20k": acc_clean,
            "All_105_Raw_Acc": acc_105_raw,
            "All_105_RentRule_Acc": acc_105_rent,
            "Non_Overlapping_96_RentRule_Acc": acc_96_rent,
            "Overlapping_9_Raw_Acc": acc_9_raw,
            "Disjoint_58_RentRule_Acc": acc_58_rent
        })

    df_summary = pd.DataFrame(eval_rows)
    out_csv = RESULTS_DIR / "leakage_removal_comparison.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Leakage removal comparison saved to {out_csv}")
    print("\n" + "=" * 70)
    print("BEFORE VS AFTER LEAKAGE REMOVAL COMPARISON TABLE:")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    # 6. McNemar Tests on New Results
    print("\n[Step 6] Running McNemar Significance Tests...")
    mc_results = {}

    # SVM Original vs Leakage-Free on 105
    mc_svm = mcnemar_test(
        df_own["category"].tolist(),
        preds_dict["SVM (Original Seed)"]["p105"],
        preds_dict["SVM (Leakage-Free)"]["p105"]
    )
    mc_results["SVM_Orig_vs_LeakageFree_105"] = mc_svm
    print(f"  SVM Orig vs Leakage-Free (N=105): chi2={mc_svm['statistic']}, p={mc_svm['p_value']}")

    # LightGBM Original vs Leakage-Free on 105
    mc_lgb = mcnemar_test(
        df_own["category"].tolist(),
        preds_dict["LightGBM (Original Seed)"]["p105"],
        preds_dict["LightGBM (Leakage-Free)"]["p105"]
    )
    mc_results["LightGBM_Orig_vs_LeakageFree_105"] = mc_lgb
    print(f"  LightGBM Orig vs Leakage-Free (N=105): chi2={mc_lgb['statistic']}, p={mc_lgb['p_value']}")

    # SVM vs LightGBM on Leakage-Free 96
    mc_svm_lgb_96 = mcnemar_test(
        df_own_96["category"].tolist(),
        preds_dict["SVM (Leakage-Free)"]["p96"],
        preds_dict["LightGBM (Leakage-Free)"]["p96"]
    )
    mc_results["SVM_vs_LightGBM_LeakageFree_96"] = mc_svm_lgb_96
    print(f"  SVM vs LightGBM on Non-Overlapping (N=96): chi2={mc_svm_lgb_96['statistic']}, p={mc_svm_lgb_96['p_value']}")

    # LightGBM vs MiniLM on Leakage-Free 96
    mc_lgb_mini_96 = mcnemar_test(
        df_own_96["category"].tolist(),
        preds_dict["LightGBM (Leakage-Free)"]["p96"],
        preds_dict["MiniLM+LR (Leakage-Free)"]["p96"]
    )
    mc_results["LightGBM_vs_MiniLM_LeakageFree_96"] = mc_lgb_mini_96
    print(f"  LightGBM vs MiniLM on Non-Overlapping (N=96): chi2={mc_lgb_mini_96['statistic']}, p={mc_lgb_mini_96['p_value']}")

    # XGBoost Standard vs XGBoost ES (100 trees) on Clean Test (N=20,040)
    mc_xgb_clean = mcnemar_test(
        y_test,
        preds_dict["XGBoost (Original Seed)"]["p_clean"],
        preds_dict["XGBoost (ES 100 Trees)"]["p_clean"]
    )
    mc_results["XGBoost_Orig_vs_ES100_CleanTest"] = mc_xgb_clean
    print(f"  XGBoost Orig vs ES 100 Trees on Clean Test (N=20,040): chi2={mc_xgb_clean['statistic']}, p={mc_xgb_clean['p_value']}")

    with open(RESULTS_DIR / "mcnemar_tests_phase5.json", "w") as f:
        json.dump(mc_results, f, indent=2)
    print(f"[SAVED] Saved McNemar test results to {RESULTS_DIR / 'mcnemar_tests_phase5.json'}")

if __name__ == "__main__":
    main()
