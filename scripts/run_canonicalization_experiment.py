"""Controlled Experiment: Transaction Canonicalization Evaluation.

Compares Logistic Regression + TF-IDF under three controlled conditions:
  Condition A: Raw transaction descriptions.
  Condition B: Canonicalized transaction descriptions.
  Condition C: Canonicalized descriptions + Engineered features (Word TF-IDF + Handcrafted).

Strict leakage safeguards:
  - Vectorizers and scalers fitted exclusively on training data.
  - Test sets (clean and noisy) evaluated out-of-sample with zero parameter tuning.
  - Merchant-grouped separation preserved.
  - Validation set used for harmful normalization audit.
"""

import sys
import time
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize

SPLIT_PATH = Path("data/mitulshah_split.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Synthetic noise transformations matching project specification
TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]
TEST_KINDS  = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]


def apply_single_noise(text: str, rng: random.Random, kind: str) -> str:
    """Applies a specific synthetic Indian banking noise transformation."""
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


def extract_handcrafted_features(texts: List[str]) -> np.ndarray:
    """Extract dense morphological and keyword features developed in Phase 3."""
    feats = []
    for t in texts:
        s = str(t)
        l = len(s)
        low = s.lower()
        d_ratio = sum(c.isdigit() for c in s) / max(l, 1)
        u_ratio = sum(c.isupper() for c in s) / max(l, 1)
        has_upi = 1.0 if "upi" in low else 0.0
        has_neft = 1.0 if "neft" in low else 0.0
        has_pos = 1.0 if "pos" in low else 0.0
        has_imps = 1.0 if "imps" in low else 0.0
        feats.append([float(l), float(d_ratio), float(u_ratio), has_upi, has_neft, has_pos, has_imps])
    return np.array(feats, dtype=np.float32)


def measure_single_item_latency(
    model: LogisticRegression,
    transform_fn,
    sample_texts: List[str],
    n_warmup: int = 200,
    n_runs: int = 1000,
    repeats: int = 5
) -> Tuple[float, float]:
    """Measures single-transaction end-to-end inference latency (preprocessing + transform + predict)."""
    # Warmup
    for i in range(n_warmup):
        txt = sample_texts[i % len(sample_texts)]
        feat = transform_fn([txt])
        _ = model.predict_proba(feat)

    latencies_per_repeat = []
    for r in range(repeats):
        t0 = time.perf_counter()
        for i in range(n_runs):
            txt = sample_texts[i % len(sample_texts)]
            feat = transform_fn([txt])
            _ = model.predict_proba(feat)
        t_total = time.perf_counter() - t0
        ms_per_txn = (t_total / n_runs) * 1000.0
        latencies_per_repeat.append(ms_per_txn)

    return float(np.mean(latencies_per_repeat)), float(np.std(latencies_per_repeat))


def main():
    print("=" * 80)
    print("TRANSACTION CANONICALIZATION CONTROLLED EXPERIMENT")
    print("=" * 80)

    # 1. Load data
    print("\n[Step 1] Loading split data from", SPLIT_PATH)
    df = pd.read_csv(SPLIT_PATH)
    train_df = df[df.split == "train"].copy().reset_index(drop=True)
    val_df   = df[df.split == "val"].copy().reset_index(drop=True)
    test_df  = df[df.split == "test"].copy().reset_index(drop=True)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Label encoding
    le = LabelEncoder().fit(df.category)
    classes = list(le.classes_)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)
    y_te_clean = le.transform(test_df.category)

    # Prepare noisy training data (35% standard augmentation, matching baseline)
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train_df)) < 0.35
    train_noisy_texts = []
    for is_noisy, t in zip(m_tr, train_df.transaction_description):
        train_noisy_texts.append(apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) if is_noisy else t)

    # Prepare noisy validation data (35% augmentation with distinct seed 11)
    r_va = random.Random(11)
    m_va = np.random.RandomState(1).rand(len(val_df)) < 0.35
    val_noisy_texts = []
    for is_noisy, t in zip(m_va, val_df.transaction_description):
        val_noisy_texts.append(apply_single_noise(t, r_va, r_va.choice(TRAIN_KINDS)) if is_noisy else t)

    # Prepare noisy test data (100% corrupted across all 8 test noise types with seed 7)
    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
    test_noisy_texts = [
        apply_single_noise(t, r_te, nt) for t, nt in zip(test_df.transaction_description, test_noise_types)
    ]
    test_df["noise_type"] = test_noise_types
    y_te_noisy = le.transform(test_df.category)

    # Prepare canonicalized text variants
    print("[Step 2] Canonicalizing text series...")
    t0_c = time.time()
    train_canon_texts = [canonicalize(t) for t in train_noisy_texts]
    val_canon_texts = [canonicalize(t) for t in val_noisy_texts]
    val_clean_canon_texts = [canonicalize(t) for t in val_df.transaction_description]
    test_clean_canon_texts = [canonicalize(t) for t in test_df.transaction_description]
    test_noisy_canon_texts = [canonicalize(t) for t in test_noisy_texts]
    print(f"  Canonicalized {len(train_canon_texts) + len(test_noisy_canon_texts):,} descriptions in {time.time() - t0_c:.2f}s.")

    # -------------------------------------------------------------------------
    # VARIANT A: Raw Transaction Descriptions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("VARIANT A: Raw Transaction Descriptions (Baseline)")
    print("-" * 70)
    tfidf_a = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=False
    )
    t0_fit = time.time()
    X_tr_a = tfidf_a.fit_transform(train_noisy_texts)
    fit_time_a = time.time() - t0_fit
    print(f"  TF-IDF fitted on training set: {X_tr_a.shape[1]} features in {fit_time_a:.2f}s.")

    X_va_a = tfidf_a.transform(val_noisy_texts)
    X_te_clean_a = tfidf_a.transform(test_df.transaction_description)
    X_te_noisy_a = tfidf_a.transform(test_noisy_texts)

    clf_a = LogisticRegression(C=10, max_iter=300, random_state=42)
    t0_tr = time.time()
    clf_a.fit(X_tr_a, y_tr)
    train_time_a = time.time() - t0_tr
    print(f"  Logistic Regression trained in {train_time_a:.2f}s.")

    pred_clean_a = clf_a.predict(X_te_clean_a)
    pred_noisy_a = clf_a.predict(X_te_noisy_a)
    acc_clean_a = accuracy_score(y_te_clean, pred_clean_a)
    f1_clean_a  = f1_score(y_te_clean, pred_clean_a, average="macro")
    acc_noisy_a = accuracy_score(y_te_noisy, pred_noisy_a)
    f1_noisy_a  = f1_score(y_te_noisy, pred_noisy_a, average="macro")

    # Latency measurement
    def transform_a(texts):
        return tfidf_a.transform(texts)

    lat_a_mean, lat_a_std = measure_single_item_latency(clf_a, transform_a, test_df.transaction_description.tolist()[:1000])
    print(f"  Clean Test: Acc = {acc_clean_a:.4f}, Macro-F1 = {f1_clean_a:.4f}")
    print(f"  Noisy Test: Acc = {acc_noisy_a:.4f}, Macro-F1 = {f1_noisy_a:.4f}")
    print(f"  Latency (Single Item): {lat_a_mean:.4f} ± {lat_a_std:.4f} ms/txn")

    # -------------------------------------------------------------------------
    # VARIANT B: Canonicalized Transaction Descriptions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("VARIANT B: Canonicalized Transaction Descriptions")
    print("-" * 70)
    tfidf_b = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=False
    )
    t0_fit = time.time()
    X_tr_b = tfidf_b.fit_transform(train_canon_texts)
    fit_time_b = time.time() - t0_fit
    print(f"  TF-IDF fitted on canonicalized training set: {X_tr_b.shape[1]} features in {fit_time_b:.2f}s.")

    X_va_b = tfidf_b.transform(val_canon_texts)
    X_te_clean_b = tfidf_b.transform(test_clean_canon_texts)
    X_te_noisy_b = tfidf_b.transform(test_noisy_canon_texts)

    clf_b = LogisticRegression(C=10, max_iter=300, random_state=42)
    t0_tr = time.time()
    clf_b.fit(X_tr_b, y_tr)
    train_time_b = time.time() - t0_tr
    print(f"  Logistic Regression trained in {train_time_b:.2f}s.")

    pred_clean_b = clf_b.predict(X_te_clean_b)
    pred_noisy_b = clf_b.predict(X_te_noisy_b)
    acc_clean_b = accuracy_score(y_te_clean, pred_clean_b)
    f1_clean_b  = f1_score(y_te_clean, pred_clean_b, average="macro")
    acc_noisy_b = accuracy_score(y_te_noisy, pred_noisy_b)
    f1_noisy_b  = f1_score(y_te_noisy, pred_noisy_b, average="macro")

    # Latency measurement (includes canonicalize() call)
    def transform_b(texts):
        return tfidf_b.transform([canonicalize(t) for t in texts])

    lat_b_mean, lat_b_std = measure_single_item_latency(clf_b, transform_b, test_df.transaction_description.tolist()[:1000])
    print(f"  Clean Test: Acc = {acc_clean_b:.4f}, Macro-F1 = {f1_clean_b:.4f}")
    print(f"  Noisy Test: Acc = {acc_noisy_b:.4f}, Macro-F1 = {f1_noisy_b:.4f}")
    print(f"  Latency (Single Item): {lat_b_mean:.4f} ± {lat_b_std:.4f} ms/txn")

    # -------------------------------------------------------------------------
    # VARIANT C: Canonicalized + Engineered Features (Word TF-IDF + Handcrafted)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("VARIANT C: Canonicalized + Engineered Features")
    print("-" * 70)
    # Additional word-level vectorizer
    tfidf_c_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=5000,
        sublinear_tf=True
    )
    t0_c_feat = time.time()
    X_tr_word = tfidf_c_word.fit_transform(train_canon_texts)
    X_va_word = tfidf_c_word.transform(val_canon_texts)
    X_te_clean_word = tfidf_c_word.transform(test_clean_canon_texts)
    X_te_noisy_word = tfidf_c_word.transform(test_noisy_canon_texts)

    # Handcrafted morphological features (7 features)
    F_tr = extract_handcrafted_features(train_canon_texts)
    F_va = extract_handcrafted_features(val_canon_texts)
    F_te_clean = extract_handcrafted_features(test_clean_canon_texts)
    F_te_noisy = extract_handcrafted_features(test_noisy_canon_texts)

    scaler = MaxAbsScaler()
    F_tr_scaled = csr_matrix(scaler.fit_transform(F_tr))
    F_va_scaled = csr_matrix(scaler.transform(F_va))
    F_te_clean_scaled = csr_matrix(scaler.transform(F_te_clean))
    F_te_noisy_scaled = csr_matrix(scaler.transform(F_te_noisy))

    # Stack: Char TF-IDF (20,000) + Word TF-IDF (5,000) + Handcrafted Scaled (7) = 25,007 features
    X_tr_c = hstack([X_tr_b, X_tr_word, F_tr_scaled]).tocsr()
    X_va_c = hstack([X_va_b, X_va_word, F_va_scaled]).tocsr()
    X_te_clean_c = hstack([X_te_clean_b, X_te_clean_word, F_te_clean_scaled]).tocsr()
    X_te_noisy_c = hstack([X_te_noisy_b, X_te_noisy_word, F_te_noisy_scaled]).tocsr()
    print(f"  Stacked features created: {X_tr_c.shape[1]} features (Char TF-IDF 20k + Word TF-IDF 5k + Handcrafted 7) in {time.time() - t0_c_feat:.2f}s.")

    clf_c = LogisticRegression(C=10, max_iter=300, random_state=42)
    t0_tr = time.time()
    clf_c.fit(X_tr_c, y_tr)
    train_time_c = time.time() - t0_tr
    print(f"  Logistic Regression trained on stacked features in {train_time_c:.2f}s.")

    pred_clean_c = clf_c.predict(X_te_clean_c)
    pred_noisy_c = clf_c.predict(X_te_noisy_c)
    acc_clean_c = accuracy_score(y_te_clean, pred_clean_c)
    f1_clean_c  = f1_score(y_te_clean, pred_clean_c, average="macro")
    acc_noisy_c = accuracy_score(y_te_noisy, pred_noisy_c)
    f1_noisy_c  = f1_score(y_te_noisy, pred_noisy_c, average="macro")

    # Latency measurement
    def transform_c(texts):
        c_texts = [canonicalize(t) for t in texts]
        ch = tfidf_b.transform(c_texts)
        wd = tfidf_c_word.transform(c_texts)
        hc = csr_matrix(scaler.transform(extract_handcrafted_features(c_texts)))
        return hstack([ch, wd, hc]).tocsr()

    lat_c_mean, lat_c_std = measure_single_item_latency(clf_c, transform_c, test_df.transaction_description.tolist()[:1000])
    print(f"  Clean Test: Acc = {acc_clean_c:.4f}, Macro-F1 = {f1_clean_c:.4f}")
    print(f"  Noisy Test: Acc = {acc_noisy_c:.4f}, Macro-F1 = {f1_noisy_c:.4f}")
    print(f"  Latency (Single Item): {lat_c_mean:.4f} ± {lat_c_std:.4f} ms/txn")

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON TABLE & DIFFERENCES RELATIVE TO BASELINE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON ACROSS VARIANTS")
    print("=" * 80)

    comparison_records = [
        {
            "Variant": "Variant A (Raw Text Baseline)",
            "Features": "Char TF-IDF (2-5, 20k)",
            "Clean_Acc": acc_clean_a,
            "Clean_MacroF1": f1_clean_a,
            "Noisy_Acc": acc_noisy_a,
            "Noisy_MacroF1": f1_noisy_a,
            "Delta_Clean_Acc": 0.0,
            "Delta_Noisy_Acc": 0.0,
            "Delta_Noisy_MacroF1": 0.0,
            "Latency_ms_per_txn": lat_a_mean,
            "Latency_std": lat_a_std,
            "Train_Time_s": round(train_time_a, 2)
        },
        {
            "Variant": "Variant B (Canonicalized)",
            "Features": "Canonicalized + Char TF-IDF (2-5, 20k)",
            "Clean_Acc": acc_clean_b,
            "Clean_MacroF1": f1_clean_b,
            "Noisy_Acc": acc_noisy_b,
            "Noisy_MacroF1": f1_noisy_b,
            "Delta_Clean_Acc": acc_clean_b - acc_clean_a,
            "Delta_Noisy_Acc": acc_noisy_b - acc_noisy_a,
            "Delta_Noisy_MacroF1": f1_noisy_b - f1_noisy_a,
            "Latency_ms_per_txn": lat_b_mean,
            "Latency_std": lat_b_std,
            "Train_Time_s": round(train_time_b, 2)
        },
        {
            "Variant": "Variant C (Canonicalized + Engineered Feats)",
            "Features": "Canonicalized + Char TF-IDF (20k) + Word TF-IDF (5k) + Handcrafted (7)",
            "Clean_Acc": acc_clean_c,
            "Clean_MacroF1": f1_clean_c,
            "Noisy_Acc": acc_noisy_c,
            "Noisy_MacroF1": f1_noisy_c,
            "Delta_Clean_Acc": acc_clean_c - acc_clean_a,
            "Delta_Noisy_Acc": acc_noisy_c - acc_noisy_a,
            "Delta_Noisy_MacroF1": f1_noisy_c - f1_noisy_a,
            "Latency_ms_per_txn": lat_c_mean,
            "Latency_std": lat_c_std,
            "Train_Time_s": round(train_time_c, 2)
        }
    ]
    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv(RESULTS_DIR / "canonicalization_comparison_metrics.csv", index=False)
    print(comp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # RESULTS BROKEN DOWN BY NOISE TYPE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-NOISE-TYPE BREAKDOWN (TEST NOISY SET)")
    print("=" * 80)
    noise_records = []
    for nt in TEST_KINDS:
        sub_idx = [i for i, kind in enumerate(test_noise_types) if kind == nt]
        y_sub = y_te_noisy[sub_idx]
        
        acc_a_sub = accuracy_score(y_sub, pred_noisy_a[sub_idx])
        f1_a_sub  = f1_score(y_sub, pred_noisy_a[sub_idx], average="macro", zero_division=0)

        acc_b_sub = accuracy_score(y_sub, pred_noisy_b[sub_idx])
        f1_b_sub  = f1_score(y_sub, pred_noisy_b[sub_idx], average="macro", zero_division=0)

        acc_c_sub = accuracy_score(y_sub, pred_noisy_c[sub_idx])
        f1_c_sub  = f1_score(y_sub, pred_noisy_c[sub_idx], average="macro", zero_division=0)

        noise_records.append({
            "Noise_Type": nt,
            "Held_Out_From_Training": "YES (Held-Out)" if nt in ["neft", "imps"] else "NO",
            "Count": len(sub_idx),
            "Variant_A_Raw_Acc": round(acc_a_sub, 4),
            "Variant_B_Canon_Acc": round(acc_b_sub, 4),
            "Variant_C_Eng_Acc": round(acc_c_sub, 4),
            "Delta_B_minus_A": round(acc_b_sub - acc_a_sub, 4),
            "Delta_C_minus_A": round(acc_c_sub - acc_a_sub, 4),
            "Variant_A_Raw_F1": round(f1_a_sub, 4),
            "Variant_B_Canon_F1": round(f1_b_sub, 4),
            "Variant_C_Eng_F1": round(f1_c_sub, 4)
        })

    noise_df = pd.DataFrame(noise_records)
    noise_df.to_csv(RESULTS_DIR / "canonicalization_per_noise_type.csv", index=False)
    print(noise_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # PER-CLASS PRECISION, RECALL, AND F1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-CLASS EVALUATION METRICS (NOISY TEST SET)")
    print("=" * 80)
    p_a, r_a, f_a, s_a = precision_recall_fscore_support(y_te_noisy, pred_noisy_a, labels=np.arange(len(classes)))
    p_b, r_b, f_b, s_b = precision_recall_fscore_support(y_te_noisy, pred_noisy_b, labels=np.arange(len(classes)))
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_te_noisy, pred_noisy_c, labels=np.arange(len(classes)))

    per_class_records = []
    for idx, c_name in enumerate(classes):
        per_class_records.append({
            "Category": c_name,
            "Support": int(s_a[idx]),
            "A_Raw_Precision": round(p_a[idx], 4),
            "A_Raw_Recall": round(r_a[idx], 4),
            "A_Raw_F1": round(f_a[idx], 4),
            "B_Canon_Precision": round(p_b[idx], 4),
            "B_Canon_Recall": round(r_b[idx], 4),
            "B_Canon_F1": round(f_b[idx], 4),
            "C_Eng_Precision": round(p_c[idx], 4),
            "C_Eng_Recall": round(r_c[idx], 4),
            "C_Eng_F1": round(f_c[idx], 4),
            "Delta_B_minus_A_F1": round(f_b[idx] - f_a[idx], 4)
        })

    per_class_df = pd.DataFrame(per_class_records)
    per_class_df.to_csv(RESULTS_DIR / "canonicalization_per_class_metrics.csv", index=False)
    print(per_class_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 5: HARMFUL NORMALIZATION AUDIT (VALIDATION SET ONLY)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: HARMFUL NORMALIZATION AUDIT (VALIDATION SET ONLY)")
    print("=" * 80)
    pred_va_a = clf_a.predict(X_va_a)
    pred_va_b = clf_b.predict(X_va_b)

    harmful_indices = []
    beneficial_indices = []

    for i in range(len(val_df)):
        true_lbl = y_va[i]
        p_a_i = pred_va_a[i]
        p_b_i = pred_va_b[i]

        # Harmful: Raw was correct, Canonicalized was incorrect
        if p_a_i == true_lbl and p_b_i != true_lbl:
            harmful_indices.append(i)
        # Beneficial: Raw was incorrect, Canonicalized was correct
        elif p_a_i != true_lbl and p_b_i == true_lbl:
            beneficial_indices.append(i)

    print(f"Validation Records Evaluated: {len(val_df):,}")
    print(f"  Beneficial Normalizations (Raw Incorrect -> Canon Correct): {len(beneficial_indices):,} ({len(beneficial_indices)/len(val_df)*100:.2f}%)")
    print(f"  Harmful Normalizations   (Raw Correct -> Canon Incorrect): {len(harmful_indices):,} ({len(harmful_indices)/len(val_df)*100:.2f}%)")
    print(f"  Net Normalized Advantage: +{len(beneficial_indices) - len(harmful_indices):,} transactions")

    # Inspect harmful examples
    harmful_examples = []
    for idx in harmful_indices[:15]:
        raw_t = val_noisy_texts[idx]
        canon_t = val_canon_texts[idx]
        true_cat = classes[y_va[idx]]
        pred_a_cat = classes[pred_va_a[idx]]
        pred_b_cat = classes[pred_va_b[idx]]
        harmful_examples.append({
            "Validation_Index": idx,
            "Original_Noisy_Description": raw_t,
            "Canonicalized_Description": canon_t,
            "True_Category": true_cat,
            "Raw_Model_Prediction": pred_a_cat,
            "Canonicalized_Model_Prediction": pred_b_cat
        })

    harmful_df = pd.DataFrame(harmful_examples)
    harmful_df.to_csv(RESULTS_DIR / "canonicalization_harmful_examples.csv", index=False)
    print("\nSample Harmful Normalization Examples on Validation Data:")
    for h in harmful_examples[:5]:
        print(f"  [Idx {h['Validation_Index']}] True: {h['True_Category']}")
        print(f"    Raw: '{h['Original_Noisy_Description']}' -> Predicted: {h['Raw_Model_Prediction']} [CORRECT]")
        print(f"    Canon: '{h['Canonicalized_Description']}' -> Predicted: {h['Canonicalized_Model_Prediction']} [INCORRECT]")
        print()

    return {
        "comp_df": comp_df,
        "noise_df": noise_df,
        "per_class_df": per_class_df,
        "harmful_df": harmful_df
    }


if __name__ == "__main__":
    main()
