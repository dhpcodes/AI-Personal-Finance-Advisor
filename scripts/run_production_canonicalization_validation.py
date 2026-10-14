"""Comprehensive validation of production model under Legacy vs Canonicalized Preprocessing.

Evaluates models/final_model.joblib strictly without retraining, modifying weights, or altering test data.
Produces CSV artifacts and detailed evaluation summaries in results/production_canonicalization_validation/.
"""

import sys
import hashlib
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.predict import TransactionPredictor
from src.preprocessing import TextPreprocessor
from src.model_comparison_suite import with_rent_rule

OUTPUT_DIR = Path("results/production_canonicalization_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    """Compute per-class precision, recall, f1, and support."""
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return pd.DataFrame({
        "Category": classes,
        "Precision": np.round(p, 4),
        "Recall": np.round(r, 4),
        "F1_Score": np.round(f1, 4),
        "Support": s
    })


def get_noise_type(desc: str) -> str:
    """Classify natural noise type from string wrapper structure."""
    s = str(desc).strip().upper()
    if s.startswith("UPI/"): return "UPI"
    if s.startswith("POS "): return "POS"
    if s.startswith("NEFT-"): return "NEFT"
    if s.startswith("IMPS/"): return "IMPS"
    return "Clean_Indian"


def main():
    print("=" * 80)
    print("PRODUCTION PREPROCESSING EVALUATION: LEGACY V1 VS CANONICALIZED")
    print("=" * 80)

    # 1. Verify Model Hashes
    model_path = Path("models/final_model.joblib")
    backup_path = Path("models/final_model_pre_canonicalization_backup.joblib")
    legacy_backup_path = Path("models/final_model_backup.joblib")

    model_hash = get_file_sha256(model_path)
    backup_hash = get_file_sha256(backup_path) if backup_path.exists() else "NOT_FOUND"
    legacy_hash = get_file_sha256(legacy_backup_path) if legacy_backup_path.exists() else "NOT_FOUND"

    print(f"\n[Integrity Check]")
    print(f"  Current final_model.joblib SHA256: {model_hash}")
    print(f"  Backup model SHA256:               {backup_hash}")
    print(f"  Hashes Match Exactly: {model_hash == backup_hash}")

    hash_df = pd.DataFrame([{
        "File": "models/final_model.joblib",
        "SHA256": model_hash,
        "Byte_Identical_To_Backup": model_hash == backup_hash
    }])
    hash_df.to_csv(OUTPUT_DIR / "model_integrity_hash.csv", index=False)

    # Initialize predictor
    predictor = TransactionPredictor(model_path=model_path)
    prep_legacy = TextPreprocessor(use_canonicalization=False)
    prep_canon = TextPreprocessor(use_canonicalization=True)
    classes = list(predictor.classes)

    overall_metrics_records = []

    # =========================================================================
    # DATASET 1: data/own_test_indian.csv (105 Natural Indian Banking Records)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION DATASET 1: data/own_test_indian.csv (N=105, Real Indian Banking)")
    print("=" * 70)
    df_own = pd.read_csv("data/own_test_indian.csv")
    df_own["noise_type"] = df_own["transaction_description"].apply(get_noise_type)

    # Legacy inference
    predictor.preprocessor = prep_legacy
    res_leg_own = predictor.predict_batch(df_own)
    df_own["pred_legacy"] = res_leg_own["predicted_category"]
    df_own["conf_legacy"] = res_leg_own["confidence"]

    # Canonicalized inference
    predictor.preprocessor = prep_canon
    res_can_own = predictor.predict_batch(df_own)
    df_own["pred_canon"] = res_can_own["predicted_category"]
    df_own["conf_canon"] = res_can_own["confidence"]

    # With Rent Rule
    df_own["pred_legacy_rule"] = with_rent_rule(df_own["transaction_description"], df_own["pred_legacy"])
    df_own["pred_canon_rule"] = with_rent_rule(df_own["transaction_description"], df_own["pred_canon"])

    # Agreement and confidence shifts
    agree_own = (df_own["pred_legacy"] == df_own["pred_canon"]).sum()
    diff_count_own = len(df_own) - agree_own
    mean_abs_conf_own = np.abs(df_own["conf_legacy"] - df_own["conf_canon"]).mean()

    # Accuracy calculations
    acc_leg_own_model = accuracy_score(df_own["category"], df_own["pred_legacy"])
    acc_can_own_model = accuracy_score(df_own["category"], df_own["pred_canon"])
    acc_leg_own_rule = accuracy_score(df_own["category"], df_own["pred_legacy_rule"])
    acc_can_own_rule = accuracy_score(df_own["category"], df_own["pred_canon_rule"])

    f1_leg_own_rule = f1_score(df_own["category"], df_own["pred_legacy_rule"], average="macro", zero_division=0)
    f1_can_own_rule = f1_score(df_own["category"], df_own["pred_canon_rule"], average="macro", zero_division=0)

    print(f"  Rows Evaluated: {len(df_own)}")
    print(f"  Agreement Rate: {agree_own}/{len(df_own)} ({agree_own/len(df_own)*100:.2f}%)")
    print(f"  Changed Predictions: {diff_count_own}")
    print(f"  Mean Absolute Confidence Shift: {mean_abs_conf_own:.6f}")
    print(f"  Model-Only Accuracy: Legacy = {acc_leg_own_model:.4f} | Canon = {acc_can_own_model:.4f} (Diff = {acc_can_own_model - acc_leg_own_model:+.4f})")
    print(f"  Model+Rule Accuracy: Legacy = {acc_leg_own_rule:.4f} | Canon = {acc_can_own_rule:.4f} (Diff = {acc_can_own_rule - acc_leg_own_rule:+.4f})")
    print(f"  Model+Rule Macro-F1: Legacy = {f1_leg_own_rule:.4f} | Canon = {f1_can_own_rule:.4f} (Diff = {f1_can_own_rule - f1_leg_own_rule:+.4f})")

    overall_metrics_records.append({
        "Dataset": "own_test_indian.csv (Model-Only)",
        "Row_Count": len(df_own),
        "Is_Held_Out": True,
        "Has_Merchant_Leakage": False,
        "Legacy_Accuracy": round(acc_leg_own_model, 4),
        "Canonicalized_Accuracy": round(acc_can_own_model, 4),
        "Accuracy_Diff": round(acc_can_own_model - acc_leg_own_model, 4),
        "Legacy_Macro_F1": round(f1_score(df_own["category"], df_own["pred_legacy"], average="macro", zero_division=0), 4),
        "Canonicalized_Macro_F1": round(f1_score(df_own["category"], df_own["pred_canon"], average="macro", zero_division=0), 4),
        "Agreement_Rate_Pct": round(agree_own / len(df_own) * 100, 2),
        "Changed_Count": diff_count_own,
        "Mean_Abs_Conf_Change": round(mean_abs_conf_own, 6)
    })
    overall_metrics_records.append({
        "Dataset": "own_test_indian.csv (Model+RentRule)",
        "Row_Count": len(df_own),
        "Is_Held_Out": True,
        "Has_Merchant_Leakage": False,
        "Legacy_Accuracy": round(acc_leg_own_rule, 4),
        "Canonicalized_Accuracy": round(acc_can_own_rule, 4),
        "Accuracy_Diff": round(acc_can_own_rule - acc_leg_own_rule, 4),
        "Legacy_Macro_F1": round(f1_leg_own_rule, 4),
        "Canonicalized_Macro_F1": round(f1_can_own_rule, 4),
        "Agreement_Rate_Pct": round(agree_own / len(df_own) * 100, 2),
        "Changed_Count": diff_count_own,
        "Mean_Abs_Conf_Change": round(mean_abs_conf_own, 6)
    })

    # Per-noise subset breakdown for own_test_indian.csv
    subset_records = []
    for st, grp in df_own.groupby("noise_type"):
        acc_l = accuracy_score(grp["category"], grp["pred_legacy_rule"])
        acc_c = accuracy_score(grp["category"], grp["pred_canon_rule"])
        conf_shift = np.abs(grp["conf_legacy"] - grp["conf_canon"]).mean()
        chg = (grp["pred_legacy"] != grp["pred_canon"]).sum()
        subset_records.append({
            "Noise_Subset": st,
            "Sample_Count": len(grp),
            "Legacy_Accuracy": round(acc_l, 4),
            "Canonicalized_Accuracy": round(acc_c, 4),
            "Accuracy_Diff": round(acc_c - acc_l, 4),
            "Changed_Predictions": int(chg),
            "Mean_Abs_Conf_Change": round(conf_shift, 6)
        })
    df_subsets = pd.DataFrame(subset_records)
    df_subsets.to_csv(OUTPUT_DIR / "own_test_subset_breakdown.csv", index=False)

    # Per-class metrics on own_test_indian
    all_own_classes = sorted(list(df_own["category"].unique()))
    p_leg_own = compute_per_class_table(df_own["category"], df_own["pred_legacy_rule"], all_own_classes)
    p_can_own = compute_per_class_table(df_own["category"], df_own["pred_canon_rule"], all_own_classes)
    p_comp_own = pd.merge(p_leg_own, p_can_own, on=["Category", "Support"], suffixes=("_Legacy", "_Canon"))
    p_comp_own.to_csv(OUTPUT_DIR / "own_test_per_class_comparison.csv", index=False)

    # Confusion matrices
    cm_leg_own = confusion_matrix(df_own["category"], df_own["pred_legacy_rule"], labels=all_own_classes)
    cm_can_own = confusion_matrix(df_own["category"], df_own["pred_canon_rule"], labels=all_own_classes)
    pd.DataFrame(cm_leg_own, index=all_own_classes, columns=all_own_classes).to_csv(OUTPUT_DIR / "own_test_cm_legacy.csv")
    pd.DataFrame(cm_can_own, index=all_own_classes, columns=all_own_classes).to_csv(OUTPUT_DIR / "own_test_cm_canon.csv")

    # Changed predictions table (Task D)
    changed_rows = []
    for idx, row in df_own.iterrows():
        is_changed = row["pred_legacy"] != row["pred_canon"]
        leg_corr = row["pred_legacy"] == row["category"]
        can_corr = row["pred_canon"] == row["category"]
        
        if is_changed:
            effect = "HELPED" if (not leg_corr and can_corr) else ("HARMED" if (leg_corr and not can_corr) else "NEUTRAL")
        else:
            effect = "NO_DIFFERENCE"

        changed_rows.append({
            "Index": idx,
            "Original_Description": row["transaction_description"],
            "Ground_Truth_Category": row["category"],
            "Noise_Type": row["noise_type"],
            "Legacy_Prediction": row["pred_legacy"],
            "Legacy_Confidence": row["conf_legacy"],
            "Legacy_Correct": leg_corr,
            "Canonicalized_Prediction": row["pred_canon"],
            "Canonicalized_Confidence": row["conf_canon"],
            "Canonicalized_Correct": can_corr,
            "Prediction_Changed": is_changed,
            "Confidence_Change": round(row["conf_canon"] - row["conf_legacy"], 4),
            "Effect": effect
        })
    df_all_cases = pd.DataFrame(changed_rows)
    df_all_cases.to_csv(OUTPUT_DIR / "own_test_prediction_details.csv", index=False)

    df_changed_only = df_all_cases[df_all_cases["Prediction_Changed"]].copy()
    df_changed_only.to_csv(OUTPUT_DIR / "changed_predictions.csv", index=False)
    print(f"  Saved changed predictions to {OUTPUT_DIR / 'changed_predictions.csv'}")

    # =========================================================================
    # DATASET 2: data/mitulshah_split.csv (Test Split, N=20,040 Unseen Merchants)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION DATASET 2: data/mitulshah_split.csv (Test Split, N=20,040)")
    print("=" * 70)
    df_split = pd.read_csv("data/mitulshah_split.csv")
    test_ms = df_split[df_split.split == "test"].copy().reset_index(drop=True)

    # Legacy inference
    predictor.preprocessor = prep_legacy
    res_leg_ms = predictor.predict_batch(test_ms)
    pred_leg_ms = res_leg_ms["predicted_category"].tolist()
    conf_leg_ms = res_leg_ms["confidence"].tolist()

    # Canonicalized inference
    predictor.preprocessor = prep_canon
    res_can_ms = predictor.predict_batch(test_ms)
    pred_can_ms = res_can_ms["predicted_category"].tolist()
    conf_can_ms = res_can_ms["confidence"].tolist()

    agree_ms = sum(p1 == p2 for p1, p2 in zip(pred_leg_ms, pred_can_ms))
    diff_count_ms = len(test_ms) - agree_ms
    mean_abs_conf_ms = np.abs(np.array(conf_leg_ms) - np.array(conf_can_ms)).mean()

    acc_leg_ms = accuracy_score(test_ms["category"], pred_leg_ms)
    acc_can_ms = accuracy_score(test_ms["category"], pred_can_ms)
    f1_leg_ms = f1_score(test_ms["category"], pred_leg_ms, average="macro", zero_division=0)
    f1_can_ms = f1_score(test_ms["category"], pred_can_ms, average="macro", zero_division=0)

    print(f"  Rows Evaluated: {len(test_ms)}")
    print(f"  Agreement Rate: {agree_ms}/{len(test_ms)} ({agree_ms/len(test_ms)*100:.2f}%)")
    print(f"  Changed Predictions: {diff_count_ms}")
    print(f"  Mean Absolute Confidence Shift: {mean_abs_conf_ms:.6f}")
    print(f"  Clean Unseen-Merchant Accuracy: Legacy = {acc_leg_ms:.4f} | Canon = {acc_can_ms:.4f} (Diff = {acc_can_ms - acc_leg_ms:+.4f})")
    print(f"  Clean Unseen-Merchant Macro-F1: Legacy = {f1_leg_ms:.4f} | Canon = {f1_can_ms:.4f} (Diff = {f1_can_ms - f1_leg_ms:+.4f})")

    overall_metrics_records.append({
        "Dataset": "mitulshah_split.csv (Test Split)",
        "Row_Count": len(test_ms),
        "Is_Held_Out": True,
        "Has_Merchant_Leakage": False,
        "Legacy_Accuracy": round(acc_leg_ms, 4),
        "Canonicalized_Accuracy": round(acc_can_ms, 4),
        "Accuracy_Diff": round(acc_can_ms - acc_leg_ms, 4),
        "Legacy_Macro_F1": round(f1_leg_ms, 4),
        "Canonicalized_Macro_F1": round(f1_can_ms, 4),
        "Agreement_Rate_Pct": round(agree_ms / len(test_ms) * 100, 2),
        "Changed_Count": diff_count_ms,
        "Mean_Abs_Conf_Change": round(mean_abs_conf_ms, 6)
    })

    # Per-class table
    p_leg_ms = compute_per_class_table(test_ms["category"], pred_leg_ms, classes)
    p_can_ms = compute_per_class_table(test_ms["category"], pred_can_ms, classes)
    p_comp_ms = pd.merge(p_leg_ms, p_can_ms, on=["Category", "Support"], suffixes=("_Legacy", "_Canon"))
    p_comp_ms.to_csv(OUTPUT_DIR / "mitulshah_per_class_comparison.csv", index=False)

    cm_leg_ms = confusion_matrix(test_ms["category"], pred_leg_ms, labels=classes)
    pd.DataFrame(cm_leg_ms, index=classes, columns=classes).to_csv(OUTPUT_DIR / "mitulshah_cm.csv")

    # =========================================================================
    # DATASET 3: data/processed/test.parquet (Legacy Test Split, N=30,000)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION DATASET 3: data/processed/test.parquet (N=30,000)")
    print("=" * 70)
    df_pq = pd.read_parquet("data/processed/test.parquet")

    # Legacy inference
    predictor.preprocessor = prep_legacy
    res_leg_pq = predictor.predict_batch(df_pq)
    pred_leg_pq = res_leg_pq["predicted_category"].tolist()
    conf_leg_pq = res_leg_pq["confidence"].tolist()

    # Canonicalized inference
    predictor.preprocessor = prep_canon
    res_can_pq = predictor.predict_batch(df_pq)
    pred_can_pq = res_can_pq["predicted_category"].tolist()
    conf_can_pq = res_can_pq["confidence"].tolist()

    agree_pq = sum(p1 == p2 for p1, p2 in zip(pred_leg_pq, pred_can_pq))
    diff_count_pq = len(df_pq) - agree_pq
    mean_abs_conf_pq = np.abs(np.array(conf_leg_pq) - np.array(conf_can_pq)).mean()

    acc_leg_pq = accuracy_score(df_pq["category"], pred_leg_pq)
    acc_can_pq = accuracy_score(df_pq["category"], pred_can_pq)
    f1_leg_pq = f1_score(df_pq["category"], pred_leg_pq, average="macro", zero_division=0)
    f1_can_pq = f1_score(df_pq["category"], pred_can_pq, average="macro", zero_division=0)

    print(f"  Rows Evaluated: {len(df_pq)}")
    print(f"  Agreement Rate: {agree_pq}/{len(df_pq)} ({agree_pq/len(df_pq)*100:.2f}%)")
    print(f"  Changed Predictions: {diff_count_pq}")
    print(f"  Mean Absolute Confidence Shift: {mean_abs_conf_pq:.6f}")
    print(f"  Accuracy: Legacy = {acc_leg_pq:.4f} | Canon = {acc_can_pq:.4f} (Diff = {acc_can_pq - acc_leg_pq:+.4f})")
    print(f"  Macro-F1: Legacy = {f1_leg_pq:.4f} | Canon = {f1_can_pq:.4f} (Diff = {f1_can_pq - f1_leg_pq:+.4f})")

    overall_metrics_records.append({
        "Dataset": "test.parquet (Legacy V1 Split)",
        "Row_Count": len(df_pq),
        "Is_Held_Out": False,
        "Has_Merchant_Leakage": True,
        "Legacy_Accuracy": round(acc_leg_pq, 4),
        "Canonicalized_Accuracy": round(acc_can_pq, 4),
        "Accuracy_Diff": round(acc_can_pq - acc_leg_pq, 4),
        "Legacy_Macro_F1": round(f1_leg_pq, 4),
        "Canonicalized_Macro_F1": round(f1_can_pq, 4),
        "Agreement_Rate_Pct": round(agree_pq / len(df_pq) * 100, 2),
        "Changed_Count": diff_count_pq,
        "Mean_Abs_Conf_Change": round(mean_abs_conf_pq, 6)
    })

    # Save summary table
    df_summary = pd.DataFrame(overall_metrics_records)
    df_summary.to_csv(OUTPUT_DIR / "overall_metrics_comparison.csv", index=False)
    print(f"\n[Artifacts Saved in {OUTPUT_DIR}]")
    for f in OUTPUT_DIR.glob("*.csv"):
        print(f"  - {f.name}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
