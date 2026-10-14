"""Train Variant B Candidate with Leakage-Safe Indian Merchant Seed.

Strict safety constraints:
- Does NOT overwrite models/final_model.joblib or backups.
- Updates models/variant_b_candidate.joblib.
- Fits vectorizer and model exclusively on training data + indian_merchants_train_seed.csv.
- Evaluates out-of-sample on unseen merchants and held-out Indian statements.
- Saves full metrics under results/variant_b_candidate/.
"""

import sys
import time
import random
import re
import hashlib
import joblib
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.canonicalize import canonicalize
from src.predict import TransactionPredictor
from src.preprocessing import TextPreprocessor
from src.model_comparison_suite import with_rent_rule, ece_score

OUTPUT_DIR = Path("results/variant_b_candidate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODEL_PATH = MODELS_DIR / "variant_b_candidate.joblib"
PRODUCTION_MODEL_PATH = MODELS_DIR / "final_model.joblib"
BACKUP_MODEL_PATH = MODELS_DIR / "final_model_pre_canonicalization_backup.joblib"

SPLIT_PATH = Path("data/mitulshah_split.csv")
SEED_PATH = Path("data/indian_merchants_train_seed.csv")
OWN_TEST_PATH = Path("data/own_test_indian.csv")

TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]
TEST_KINDS  = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
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
    s = str(desc).strip().upper()
    if s.startswith("UPI/"): return "UPI"
    if s.startswith("POS "): return "POS"
    if s.startswith("NEFT-"): return "NEFT"
    if s.startswith("IMPS/"): return "IMPS"
    return "Clean_Indian"


def measure_single_item_latency(predictor: TransactionPredictor, texts: list, n_runs: int = 500) -> float:
    for i in range(50):
        _ = predictor.predict_single(texts[i % len(texts)])
    t0 = time.perf_counter()
    for i in range(n_runs):
        _ = predictor.predict_single(texts[i % len(texts)])
    t_tot = time.perf_counter() - t0
    return (t_tot / n_runs) * 1000.0


def main():
    print("=" * 80)
    print("TRAINING VARIANT B CANDIDATE WITH LEAKAGE-SAFE INDIAN MERCHANT SEED")
    print("=" * 80)

    # 1. Verify baseline production model hashes
    prod_hash_before = get_file_sha256(PRODUCTION_MODEL_PATH)
    backup_hash_before = get_file_sha256(BACKUP_MODEL_PATH)
    print(f"[Safety Check] Production Model SHA256: {prod_hash_before}")
    print(f"[Safety Check] Backup Model SHA256:     {backup_hash_before}")
    assert prod_hash_before == backup_hash_before, "Production model does not match backup hash!"

    # 2. Load data
    print(f"\n[Step 1] Loading splits from {SPLIT_PATH} and Indian seed from {SEED_PATH}...")
    df = pd.read_csv(SPLIT_PATH)
    train_base = df[df.split == "train"].copy().reset_index(drop=True)
    val_df     = df[df.split == "val"].copy().reset_index(drop=True)
    test_df    = df[df.split == "test"].copy().reset_index(drop=True)

    df_seed = pd.read_csv(SEED_PATH)
    print(f"  Base Train: {len(train_base):,} | Indian Seed: {len(df_seed):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Verify zero leakage between seed and own_test_indian
    df_own_check = pd.read_csv(OWN_TEST_PATH)
    overlap_seed_test = set(df_seed["transaction_description"].str.strip().str.upper()).intersection(
        set(df_own_check["transaction_description"].str.strip().str.upper())
    )
    assert len(overlap_seed_test) == 0, f"FATAL: Overlap detected: {overlap_seed_test}"
    print("  [LEAKAGE CHECK PASS] Seed has 0 overlap with held-out Indian test set.")

    # Combine base train + Indian seed strictly in train split
    train_df = pd.concat([
        train_base[["transaction_description", "category"]],
        df_seed[["transaction_description", "category"]]
    ], ignore_index=True)
    print(f"  Total Augmented Training Set Size: {len(train_df):,} rows.")

    le = LabelEncoder().fit(df.category)
    classes = list(le.classes_)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)
    y_te = le.transform(test_df.category)

    # Train noise augmentation: 35% with deterministic seed 42
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train_df)) < 0.35
    train_noisy = [
        apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) if is_n else t
        for is_n, t in zip(m_tr, train_df.transaction_description)
    ]

    # Validation noise augmentation: 35% with distinct seed 11
    r_va = random.Random(11)
    m_va = np.random.RandomState(1).rand(len(val_df)) < 0.35
    val_noisy = [
        apply_single_noise(t, r_va, r_va.choice(TRAIN_KINDS)) if is_n else t
        for is_n, t in zip(m_va, val_df.transaction_description)
    ]

    # Preprocess training and validation sets using TextPreprocessor (canonicalize -> clean_text)
    print("  Preprocessing training and validation sets using TextPreprocessor...")
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in train_noisy]
    val_clean   = [preprocessor.clean_text(t) for t in val_noisy]

    # Fit char_wb (2, 5) TF-IDF strictly on training set
    print("  Fitting TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=20000, min_df=3, lowercase=True)...")
    t0_vec = time.time()
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=True
    )
    X_tr = tfidf.fit_transform(train_clean)
    X_va = tfidf.transform(val_clean)
    vec_time = time.time() - t0_vec
    print(f"  Vectorizer fitted in {vec_time:.2f}s. Vocabulary size: {len(tfidf.vocabulary_)}")

    # Model training: LinearSVC (C=10.0) with CalibratedClassifierCV
    print("  Training CalibratedClassifierCV(LinearSVC(C=10.0, max_iter=2000), method='sigmoid')...")
    t0_fit = time.time()
    base_svc = LinearSVC(C=10.0, max_iter=2000, random_state=42)
    candidate_model = CalibratedClassifierCV(base_svc, method="sigmoid")
    candidate_model.fit(X_tr, y_tr)
    fit_time = time.time() - t0_fit
    print(f"  Model trained in {fit_time:.2f}s.")

    # Validation check
    val_preds = candidate_model.predict(X_va)
    val_acc = accuracy_score(y_va, val_preds)
    val_f1  = f1_score(y_va, val_preds, average="macro")
    print(f"  Validation Score -> Accuracy: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")

    # Save candidate model artifact
    print(f"\n[Step 2] Saving candidate model to {CANDIDATE_MODEL_PATH}...")
    candidate_artifact = {
        "version": "3.1.0-variant_b_indian_seed",
        "model_name": "Variant B Calibrated Linear SVM (char_wb 2-5 + Indian Seed)",
        "vectorizer": tfidf,
        "label_encoder": le,
        "model": candidate_model,
        "classes": classes,
        "metrics": {
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_f1), 4),
            "train_samples": len(train_df),
            "fit_time_sec": round(float(fit_time), 2)
        },
        "preprocessing_metadata": {
            "pipeline": "canonicalize -> clean_text (char_wb 2-5)",
            "vectorizer_params": tfidf.get_params(),
            "augmentation_rate": 0.35,
            "augmentation_seed": 42,
            "indian_seed_samples": len(df_seed)
        }
    }
    joblib.dump(candidate_artifact, CANDIDATE_MODEL_PATH)
    candidate_size_mb = CANDIDATE_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved candidate artifact ({candidate_size_mb:.2f} MB).")

    # -------------------------------------------------------------------------
    # EVALUATION: PRODUCTION VS RETRAINED VARIANT B CANDIDATE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: FAIR EVALUATION ON UNTOUCHED BENCHMARKS")
    print("=" * 70)

    prod_predictor = TransactionPredictor(model_path=PRODUCTION_MODEL_PATH)
    cand_predictor = TransactionPredictor(model_path=CANDIDATE_MODEL_PATH)

    # -------------------------------------------------------------------------
    # Evaluation on data/mitulshah_split.csv Test Set (20,040 unseen merchants)
    # -------------------------------------------------------------------------
    test_clean_descs = test_df.transaction_description.tolist()

    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
    test_noisy_descs = [
        apply_single_noise(t, r_te, nt) for t, nt in zip(test_clean_descs, test_noise_types)
    ]
    test_df["noise_type"] = test_noise_types

    # Predict Clean Test
    pred_prod_clean = prod_predictor.predict_batch(pd.DataFrame({"transaction_description": test_clean_descs}))
    pred_cand_clean = cand_predictor.predict_batch(pd.DataFrame({"transaction_description": test_clean_descs}))

    acc_prod_clean = accuracy_score(y_te, le.transform(pred_prod_clean["predicted_category"]))
    f1_prod_clean  = f1_score(y_te, le.transform(pred_prod_clean["predicted_category"]), average="macro")
    acc_cand_clean = accuracy_score(y_te, le.transform(pred_cand_clean["predicted_category"]))
    f1_cand_clean  = f1_score(y_te, le.transform(pred_cand_clean["predicted_category"]), average="macro")

    # Predict Noisy Test
    pred_prod_noisy = prod_predictor.predict_batch(pd.DataFrame({"transaction_description": test_noisy_descs}))
    pred_cand_noisy = cand_predictor.predict_batch(pd.DataFrame({"transaction_description": test_noisy_descs}))

    y_prod_noisy = le.transform(pred_prod_noisy["predicted_category"])
    y_cand_noisy = le.transform(pred_cand_noisy["predicted_category"])

    acc_prod_noisy = accuracy_score(y_te, y_prod_noisy)
    f1_prod_noisy  = f1_score(y_te, y_prod_noisy, average="macro")
    acc_cand_noisy = accuracy_score(y_te, y_cand_noisy)
    f1_cand_noisy  = f1_score(y_te, y_cand_noisy, average="macro")

    print(f"  Clean Test: Production Acc = {acc_prod_clean:.4f} (F1 = {f1_prod_clean:.4f}) | Candidate Acc = {acc_cand_clean:.4f} (F1 = {f1_cand_clean:.4f})")
    print(f"  Noisy Test: Production Acc = {acc_prod_noisy:.4f} (F1 = {f1_prod_noisy:.4f}) | Candidate Acc = {acc_cand_noisy:.4f} (F1 = {f1_cand_noisy:.4f})")

    # Per-noise type breakdown
    test_df["pred_prod"] = pred_prod_noisy["predicted_category"]
    test_df["pred_cand"] = pred_cand_noisy["predicted_category"]
    test_df["y_true"] = y_te

    noise_breakdown_records = []
    for nt in TEST_KINDS:
        sub = test_df[test_df.noise_type == nt]
        acc_p = accuracy_score(sub.y_true, le.transform(sub.pred_prod))
        acc_c = accuracy_score(sub.y_true, le.transform(sub.pred_cand))
        f1_c  = f1_score(sub.y_true, le.transform(sub.pred_cand), average="macro")
        noise_breakdown_records.append({
            "Noise_Type": nt,
            "Count": len(sub),
            "Production_Accuracy": round(acc_p, 4),
            "Candidate_Accuracy": round(acc_c, 4),
            "Candidate_Macro_F1": round(f1_c, 4),
            "Accuracy_Delta": round(acc_c - acc_p, 4)
        })
    df_noise_breakdown = pd.DataFrame(noise_breakdown_records)
    df_noise_breakdown.to_csv(OUTPUT_DIR / "mitulshah_noise_type_breakdown.csv", index=False)
    print("\n--- Performance by Noise Type (mitulshah_split.csv) ---")
    print(df_noise_breakdown.to_string(index=False))

    # -------------------------------------------------------------------------
    # Evaluation on data/own_test_indian.csv (105 real Indian banking transactions)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating on own_test_indian.csv (N=105) ---")
    df_own = pd.read_csv(OWN_TEST_PATH)
    df_own["noise_type"] = df_own.transaction_description.apply(get_noise_type_own)

    res_prod_own = prod_predictor.predict_batch(df_own)
    res_cand_own = cand_predictor.predict_batch(df_own)

    df_own["pred_prod"] = res_prod_own["predicted_category"]
    df_own["conf_prod"] = res_prod_own["confidence"]
    df_own["pred_cand"] = res_cand_own["predicted_category"]
    df_own["conf_cand"] = res_cand_own["confidence"]

    # Model + Rent Rule
    df_own["pred_prod_rule"] = with_rent_rule(df_own.transaction_description, df_own.pred_prod)
    df_own["pred_cand_rule"] = with_rent_rule(df_own.transaction_description, df_own.pred_cand)

    acc_prod_own_model = accuracy_score(df_own.category, df_own.pred_prod)
    acc_cand_own_model = accuracy_score(df_own.category, df_own.pred_cand)
    acc_prod_own_rule  = accuracy_score(df_own.category, df_own.pred_prod_rule)
    acc_cand_own_rule  = accuracy_score(df_own.category, df_own.pred_cand_rule)

    f1_prod_own_rule = f1_score(df_own.category, df_own.pred_prod_rule, average="macro", zero_division=0)
    f1_cand_own_rule = f1_score(df_own.category, df_own.pred_cand_rule, average="macro", zero_division=0)

    print(f"  Model-Only Accuracy: Production = {acc_prod_own_model:.4f} | Candidate = {acc_cand_own_model:.4f} (Delta = {acc_cand_own_model - acc_prod_own_model:+.4f})")
    print(f"  Model+Rule Accuracy: Production = {acc_prod_own_rule:.4f} | Candidate = {acc_cand_own_rule:.4f} (Delta = {acc_cand_own_rule - acc_prod_own_rule:+.4f})")
    print(f"  Model+Rule Macro-F1: Production = {f1_prod_own_rule:.4f} | Candidate = {f1_cand_own_rule:.4f} (Delta = {f1_cand_own_rule - f1_prod_own_rule:+.4f})")

    # Indian test subset breakdown
    own_subset_records = []
    for st, grp in df_own.groupby("noise_type"):
        acc_p = accuracy_score(grp.category, grp.pred_prod_rule)
        acc_c = accuracy_score(grp.category, grp.pred_cand_rule)
        own_subset_records.append({
            "Noise_Subset": st,
            "Count": len(grp),
            "Production_Accuracy": round(acc_p, 4),
            "Candidate_Accuracy": round(acc_c, 4),
            "Accuracy_Delta": round(acc_c - acc_p, 4)
        })
    df_own_subsets = pd.DataFrame(own_subset_records)
    df_own_subsets.to_csv(OUTPUT_DIR / "own_test_subsets_comparison.csv", index=False)
    print("\n--- Indian Banking Subsets (own_test_indian.csv) ---")
    print(df_own_subsets.to_string(index=False))

    # Consolidated benchmark summary
    lat_prod = measure_single_item_latency(prod_predictor, test_clean_descs[:200])
    lat_cand = measure_single_item_latency(cand_predictor, test_clean_descs[:200])
    prod_size_mb = PRODUCTION_MODEL_PATH.stat().st_size / (1024 * 1024)

    summary_records = [
        {"Metric": "mitulshah_clean_accuracy", "Production_Model": round(acc_prod_clean, 4), "Candidate_Variant_B": round(acc_cand_clean, 4), "Delta": round(acc_cand_clean - acc_prod_clean, 4)},
        {"Metric": "mitulshah_clean_macro_f1", "Production_Model": round(f1_prod_clean, 4), "Candidate_Variant_B": round(f1_cand_clean, 4), "Delta": round(f1_cand_clean - f1_prod_clean, 4)},
        {"Metric": "mitulshah_noisy_accuracy", "Production_Model": round(acc_prod_noisy, 4), "Candidate_Variant_B": round(acc_cand_noisy, 4), "Delta": round(acc_cand_noisy - acc_prod_noisy, 4)},
        {"Metric": "mitulshah_noisy_macro_f1", "Production_Model": round(f1_prod_noisy, 4), "Candidate_Variant_B": round(f1_cand_noisy, 4), "Delta": round(f1_cand_noisy - f1_prod_noisy, 4)},
        {"Metric": "mitulshah_typo_accuracy", "Production_Model": round(df_noise_breakdown[df_noise_breakdown.Noise_Type == "typo"]["Production_Accuracy"].values[0], 4), "Candidate_Variant_B": round(df_noise_breakdown[df_noise_breakdown.Noise_Type == "typo"]["Candidate_Accuracy"].values[0], 4), "Delta": round(df_noise_breakdown[df_noise_breakdown.Noise_Type == "typo"]["Accuracy_Delta"].values[0], 4)},
        {"Metric": "own_test_indian_model_only_acc", "Production_Model": round(acc_prod_own_model, 4), "Candidate_Variant_B": round(acc_cand_own_model, 4), "Delta": round(acc_cand_own_model - acc_prod_own_model, 4)},
        {"Metric": "own_test_indian_model_rent_rule_acc", "Production_Model": round(acc_prod_own_rule, 4), "Candidate_Variant_B": round(acc_cand_own_rule, 4), "Delta": round(acc_cand_own_rule - acc_prod_own_rule, 4)},
        {"Metric": "inference_latency_ms_per_txn", "Production_Model": round(lat_prod, 4), "Candidate_Variant_B": round(lat_cand, 4), "Delta": round(lat_cand - lat_prod, 4)},
        {"Metric": "model_artifact_size_mb", "Production_Model": round(prod_size_mb, 2), "Candidate_Variant_B": round(candidate_size_mb, 2), "Delta": round(candidate_size_mb - prod_size_mb, 2)}
    ]
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(OUTPUT_DIR / "production_vs_candidate_benchmark.csv", index=False)
    print("\n=== CONSOLIDATED BENCHMARK SUMMARY ===")
    print(df_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # TARGET INDIAN MERCHANT CASES EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: TARGET INDIAN MERCHANT CASES EVALUATION")
    print("=" * 70)

    target_cases = [
        {"merchant": "Amazon Pay", "raw_desc": "UPI/998124019283/AMAZON PAY/amazon@apl", "expected": "Shopping & Retail"},
        {"merchant": "Swiggy", "raw_desc": "UPI/308942189012/SWIGGY/swiggy@icici", "expected": "Food & Dining"},
        {"merchant": "Zomato", "raw_desc": "UPI/489120938102/ZOMATO/zomato@hdfcbank", "expected": "Food & Dining"},
        {"merchant": "Ola", "raw_desc": "UPI/992019283102/OLA CABS/olacabs@icici", "expected": "Transportation"},
        {"merchant": "Jio", "raw_desc": "UPI/110293849102/RELIANCE JIO RECHARGE/jio@paytm", "expected": "Utilities & Services"},
        {"merchant": "Blinkit", "raw_desc": "BLINKIT COMMERCE PVT LTD", "expected": "Food & Dining"},
        {"merchant": "Zepto", "raw_desc": "ZEPTO 10 MIN GROCERY DELIVERY", "expected": "Food & Dining"},
        {"merchant": "BigBasket", "raw_desc": "BIGBASKET SUPERMARKET ORDER", "expected": "Food & Dining"},
        {"merchant": "Flipkart", "raw_desc": "IMPS/3948201/FLIPKART INTERNET/REF921", "expected": "Shopping & Retail"},
        {"merchant": "Myntra", "raw_desc": "POS 5241XXXX9021 MYNTRA DESIGNS BANGALORE", "expected": "Shopping & Retail"},
        {"merchant": "Airtel", "raw_desc": "BHARTI AIRTEL POSTPAID BILL PAYMENT", "expected": "Utilities & Services"}
    ]

    merchant_case_records = []
    for tc in target_cases:
        desc = tc["raw_desc"]
        exp = tc["expected"]
        m_name = tc["merchant"]

        in_train_count = train_df.transaction_description.str.contains(m_name, case=False, na=False).sum()

        res_p = prod_predictor.predict_single(desc)
        res_c = cand_predictor.predict_single(desc)

        pred_p = res_p["predicted_category"]
        conf_p = res_p["confidence"]
        corr_p = pred_p == exp

        pred_c = res_c["predicted_category"]
        conf_c = res_c["confidence"]
        corr_c = pred_c == exp

        merchant_case_records.append({
            "Merchant": m_name,
            "Input_Transaction": desc,
            "Expected_Category": exp,
            "Present_In_Train_Set": in_train_count > 0,
            "Train_Occurrences": int(in_train_count),
            "Production_Pred": pred_p,
            "Production_Conf": conf_p,
            "Production_Correct": corr_p,
            "Candidate_Pred": pred_c,
            "Candidate_Conf": conf_c,
            "Candidate_Correct": corr_c
        })

    df_merchant_cases = pd.DataFrame(merchant_case_records)
    df_merchant_cases.to_csv(OUTPUT_DIR / "indian_merchant_evaluation.csv", index=False)
    print("\n--- Indian Merchant Specific Results ---")
    print(df_merchant_cases.to_string(index=False))

    # -------------------------------------------------------------------------
    # ARTIFACT INTEGRITY CHECK
    # -------------------------------------------------------------------------
    prod_hash_after = get_file_sha256(PRODUCTION_MODEL_PATH)
    backup_hash_after = get_file_sha256(BACKUP_MODEL_PATH)
    cand_hash = get_file_sha256(CANDIDATE_MODEL_PATH)

    print("\n" + "=" * 70)
    print("STEP 5: ARTIFACT INTEGRITY & ISOLATION CHECK")
    print("=" * 70)
    print(f"Production Model SHA256 (Before): {prod_hash_before}")
    print(f"Production Model SHA256 (After):  {prod_hash_after}")
    print(f"Backup Model SHA256 (After):      {backup_hash_after}")
    print(f"Candidate Model SHA256:           {cand_hash}")

    assert prod_hash_before == prod_hash_after, "FATAL: Production model hash changed!"
    assert backup_hash_before == backup_hash_after, "FATAL: Backup model hash changed!"
    print("[PASS] Production and backup models are completely unmodified and byte-identical.")

    integrity_df = pd.DataFrame([
        {"File": str(PRODUCTION_MODEL_PATH), "SHA256": prod_hash_after, "Status": "UNTOUCHED_AND_VERIFIED"},
        {"File": str(BACKUP_MODEL_PATH), "SHA256": backup_hash_after, "Status": "UNTOUCHED_AND_VERIFIED"},
        {"File": str(CANDIDATE_MODEL_PATH), "SHA256": cand_hash, "Status": "NEW_CANDIDATE_ARTIFACT"}
    ])
    integrity_df.to_csv(OUTPUT_DIR / "artifact_hashes.csv", index=False)
    print(f"\nAll evaluation artifacts successfully saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
