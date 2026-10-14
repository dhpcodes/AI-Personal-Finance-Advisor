"""
Evaluate DistilBERT Predictions from Colab and append to results/model_comparison.csv.
Script: scripts/eval_distilbert_predictions.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_comparison_suite import with_rent_rule

PRED_DIR = REPO_ROOT / "results" / "distilbert"
COMPARISON_CSV = REPO_ROOT / "results" / "model_comparison.csv"

def main():
    print("=" * 70)
    print("DISTILBERT PREDICTION EVALUATION")
    print("=" * 70)

    clean_csv = PRED_DIR / "distilbert_preds_clean_test.csv"
    noisy_csv = PRED_DIR / "distilbert_preds_noisy_test.csv"
    typo_csv  = PRED_DIR / "distilbert_preds_typo_test.csv"
    own_csv   = PRED_DIR / "distilbert_preds_own_test_indian.csv"

    missing = [str(p) for p in [clean_csv, noisy_csv, typo_csv, own_csv] if not p.exists()]
    if missing:
        print("\n[STATUS: NOT RUN]")
        print("DistilBERT predictions not found locally. Missing files:")
        for m in missing:
            print(f"  - {m}")
        print("\nTo evaluate DistilBERT:")
        print("1. Upload notebooks/05_distilbert_colab.py to Google Colab with data files.")
        print("2. Run the notebook on a free T4 GPU.")
        print("3. Download the 4 prediction CSVs into 'results/distilbert/'.")
        print("4. Re-run this script to automatically update results/model_comparison.csv.")
        return

    print("Loading prediction CSVs...")
    df_clean = pd.read_csv(clean_csv)
    df_noisy = pd.read_csv(noisy_csv)
    df_typo  = pd.read_csv(typo_csv)
    df_own   = pd.read_csv(own_csv)

    clean_acc = accuracy_score(df_clean["true_category"], df_clean["predicted_category"])
    clean_macro_f1 = f1_score(df_clean["true_category"], df_clean["predicted_category"], average="macro", zero_division=0)
    clean_weighted_f1 = f1_score(df_clean["true_category"], df_clean["predicted_category"], average="weighted", zero_division=0)

    noisy_acc = accuracy_score(df_noisy["true_category"], df_noisy["predicted_category"])
    noisy_macro_f1 = f1_score(df_noisy["true_category"], df_noisy["predicted_category"], average="macro", zero_division=0)
    noisy_weighted_f1 = f1_score(df_noisy["true_category"], df_noisy["predicted_category"], average="weighted", zero_division=0)

    typo_acc = accuracy_score(df_typo["true_category"], df_typo["predicted_category"])

    own_model_acc = accuracy_score(df_own["true_category"], df_own["predicted_category"])
    own_rule_preds = with_rent_rule(df_own["transaction_description"], df_own["predicted_category"])
    own_rule_acc = accuracy_score(df_own["true_category"], own_rule_preds)
    own_rule_f1 = f1_score(df_own["true_category"], own_rule_preds, average="macro", zero_division=0)

    print(f"Clean Test Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    print(f"Clean Test Macro-F1: {clean_macro_f1:.4f}")
    print(f"Noisy Test Accuracy: {noisy_acc:.4f} ({noisy_acc*100:.2f}%)")
    print(f"Typo Noise Accuracy: {typo_acc:.4f} ({typo_acc*100:.2f}%)")
    print(f"Indian Statements Model-Only: {own_model_acc:.4f} ({own_model_acc*100:.2f}%)")
    print(f"Indian Statements Model+Rent: {own_rule_acc:.4f} ({own_rule_acc*100:.2f}%)")

    distilbert_row = {
        "Model": "DistilBERT",
        "Validation_Accuracy": None,
        "Validation_Macro_F1": None,
        "Clean_Test_Accuracy": round(clean_acc, 4),
        "Clean_Test_Macro_F1": round(clean_macro_f1, 4),
        "Clean_Test_Weighted_F1": round(clean_weighted_f1, 4),
        "Noisy_Test_Accuracy": round(noisy_acc, 4),
        "Noisy_Test_Macro_F1": round(noisy_macro_f1, 4),
        "Noisy_Test_Weighted_F1": round(noisy_weighted_f1, 4),
        "Typo_Accuracy": round(typo_acc, 4),
        "Indian_Statement_Model_Acc": round(own_model_acc, 4),
        "Indian_Statement_Model_Plus_RentRule_Acc": round(own_rule_acc, 4),
        "Indian_Statement_Macro_F1": round(own_rule_f1, 4),
        "Training_Time_Sec": None,
        "Inference_Latency_ms_per_txn": None,
        "Artifact_Size_MB": None
    }

    if COMPARISON_CSV.exists():
        df_comp = pd.read_csv(COMPARISON_CSV)
        # Drop existing DistilBERT row if present
        df_comp = df_comp[df_comp["Model"] != "DistilBERT"]
        df_comp = pd.concat([df_comp, pd.DataFrame([distilbert_row])], ignore_index=True)
        df_comp.to_csv(COMPARISON_CSV, index=False)
        print(f"\nAppended DistilBERT metrics to {COMPARISON_CSV}.")

if __name__ == "__main__":
    main()
