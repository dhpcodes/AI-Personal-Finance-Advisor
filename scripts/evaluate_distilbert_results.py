"""Evaluate DistilBERT Fine-Tuned Predictions and Compute Metrics & Statistical Tests.

Tasks:
1. Verify alignment (20,040 clean, 20,040 noisy v2, 105 Indian).
2. Compute Accuracy, Macro-F1, Wilson 95% CI, and ECE for clean test, noisy v2, and Indian sets.
3. Compute Indian MODEL-ONLY accuracy for N=105, N=96, and N=58.
4. Compute per-noise-type accuracy on Noisy V2.
5. Run McNemar tests:
   - vs Linear SVM (20k) on clean test and noisy v2
   - vs Matched Linear SVM (3k) on clean test and noisy v2
   - vs MiniLM and LightGBM on Indian N=96 subset
6. Append results to results/final_metrics.csv under a new row.
7. Save detailed breakdown to results/distilbert_evaluation_summary.csv and mcnemar_tests_distilbert.json.
"""

import sys
import time
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from src.preprocessing import TextPreprocessor
from src.predict import TransactionPredictor
from src.model_comparison_suite import with_rent_rule

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

CLEAN_PREDS_PATH = RESULTS_DIR / "distilbert_preds_clean_test.csv"
NOISY_PREDS_PATH = RESULTS_DIR / "distilbert_preds_noisy_v2.csv"
INDIAN_PREDS_PATH = RESULTS_DIR / "distilbert_preds_indian_105.csv"

SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
NOISY_V2_PATH = DATA_DIR / "noisy_test_v2.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"
CLEAN_SEED_PATH = DATA_DIR / "indian_merchants_train_seed_leakage_free.csv"
FINAL_METRICS_PATH = RESULTS_DIR / "final_metrics.csv"

OVERLAPPING_CORES = [
    "amazon pay", "dominos pizza", "flipkart internet", "myntra designs",
    "ola cabs", "swiggy", "uber india", "zomato"
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

def compute_ece(true_labels, pred_labels, confidences, bins: int = 15) -> float:
    confs = np.asarray(confidences, dtype=float)
    correct = np.asarray([yt == yp for yt, yp in zip(true_labels, pred_labels)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total_n = len(confs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(correct[mask])
            bin_conf = np.mean(confs[mask])
            ece += (bin_count / total_n) * abs(bin_acc - bin_conf)
    return float(ece)

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
    print("=" * 75)
    print("DISTILBERT EVALUATION & STATISTICAL COMPARISON")
    print("=" * 75)

    # 1. Load Prediction Files & Ground Truth
    df_clean = pd.read_csv(CLEAN_PREDS_PATH)
    df_noisy = pd.read_csv(NOISY_PREDS_PATH)
    df_ind = pd.read_csv(INDIAN_PREDS_PATH)

    df_split = pd.read_csv(SPLIT_PATH)
    test_split = df_split[df_split.split == "test"].reset_index(drop=True)
    df_nv2_gt = pd.read_csv(NOISY_V2_PATH).reset_index(drop=True)
    df_own_gt = pd.read_csv(OWN_TEST_PATH).reset_index(drop=True)

    # Strict Alignment Verification
    assert len(df_clean) == 20040 and (df_clean["raw_text"] == test_split["transaction_description"]).all()
    assert len(df_noisy) == 20040 and (df_noisy["raw_text"] == df_nv2_gt["transaction_description"]).all()
    assert len(df_ind) == 105 and (df_ind["raw_text"] == df_own_gt["transaction_description"]).all()
    print(">> [ALIGNMENT CHECK PASS] All 3 prediction sets match ground truth text 100% row-by-row.\n")

    # 2. Clean Test Evaluation
    clean_acc = accuracy_score(df_clean["true_label"], df_clean["pred_label"])
    clean_f1 = f1_score(df_clean["true_label"], df_clean["pred_label"], average="macro")
    c_ci_low, c_ci_high = wilson_score_interval(clean_acc, len(df_clean))
    clean_ece = compute_ece(df_clean["true_label"], df_clean["pred_label"], df_clean["confidence"])

    print(f"Clean Test (N=20,040):")
    print(f"  Accuracy:  {clean_acc*100:.2f}% [{c_ci_low*100:.2f}%, {c_ci_high*100:.2f}%]")
    print(f"  Macro-F1:  {clean_f1:.4f}")
    print(f"  ECE:       {clean_ece:.4f}")

    # 3. Noisy V2 Evaluation
    noisy_acc = accuracy_score(df_noisy["true_label"], df_noisy["pred_label"])
    noisy_f1 = f1_score(df_noisy["true_label"], df_noisy["pred_label"], average="macro")
    n_ci_low, n_ci_high = wilson_score_interval(noisy_acc, len(df_noisy))
    noisy_ece = compute_ece(df_noisy["true_label"], df_noisy["pred_label"], df_noisy["confidence"])

    print(f"\nNoisy Test V2 (N=20,040):")
    print(f"  Accuracy:  {noisy_acc*100:.2f}% [{n_ci_low*100:.2f}%, {n_ci_high*100:.2f}%]")
    print(f"  Macro-F1:  {noisy_f1:.4f}")
    print(f"  ECE:       {noisy_ece:.4f}")

    # Per-Noise-Type Accuracy on Noisy V2
    print("\nPer-Noise-Type Accuracy on Noisy V2:")
    noise_breakdown = []
    for ntype, grp in df_noisy.groupby("noise_type"):
        nt_acc = accuracy_score(grp["true_label"], grp["pred_label"])
        nt_low, nt_high = wilson_score_interval(nt_acc, len(grp))
        noise_breakdown.append({
            "noise_type": ntype,
            "count": len(grp),
            "accuracy": round(nt_acc, 4),
            "ci_95": f"[{nt_low*100:.2f}%, {nt_high*100:.2f}%]"
        })
        print(f"  {ntype:12} (N={len(grp):,}): {nt_acc*100:.2f}% [{nt_low*100:.2f}%, {nt_high*100:.2f}%]")

    # 4. Indian Statements Evaluation (N=105, N=96, N=58)
    df_ind["canon"] = df_ind["raw_text"].apply(canonicalize)
    is_ov_9 = df_ind["canon"].apply(lambda c: any(core in c.lower() for core in OVERLAPPING_CORES))
    df_ind_96 = df_ind[~is_ov_9].copy()

    df_clean_seed = pd.read_csv(CLEAN_SEED_PATH)
    clean_seed_cores = set(df_clean_seed["core"].dropna().str.lower().unique())
    is_in_clean_seed = df_ind["canon"].apply(lambda c: any(sc in c.lower() or c.lower() in sc for sc in clean_seed_cores))
    df_ind_58 = df_ind[~is_in_clean_seed].copy()

    # Model-Only Accuracy (Primary)
    ind_105_raw_acc = accuracy_score(df_ind["true_label"], df_ind["pred_label"])
    ind_105_raw_f1 = f1_score(df_ind["true_label"], df_ind["pred_label"], average="macro")
    i105_ci_low, i105_ci_high = wilson_score_interval(ind_105_raw_acc, len(df_ind))
    ind_105_ece = compute_ece(df_ind["true_label"], df_ind["pred_label"], df_ind["confidence"])

    ind_96_raw_acc = accuracy_score(df_ind_96["true_label"], df_ind_96["pred_label"])
    i96_ci_low, i96_ci_high = wilson_score_interval(ind_96_raw_acc, len(df_ind_96))

    ind_58_raw_acc = accuracy_score(df_ind_58["true_label"], df_ind_58["pred_label"])
    i58_ci_low, i58_ci_high = wilson_score_interval(ind_58_raw_acc, len(df_ind_58))

    # +Rent Rule Accuracy (Secondary)
    p105_rent = with_rent_rule(df_ind["raw_text"].tolist(), df_ind["pred_label"].tolist())
    ind_105_rent_acc = accuracy_score(df_ind["true_label"], p105_rent)
    i105_rent_low, i105_rent_high = wilson_score_interval(ind_105_rent_acc, len(df_ind))

    p96_rent = with_rent_rule(df_ind_96["raw_text"].tolist(), df_ind_96["pred_label"].tolist())
    ind_96_rent_acc = accuracy_score(df_ind_96["true_label"], p96_rent)

    p58_rent = with_rent_rule(df_ind_58["raw_text"].tolist(), df_ind_58["pred_label"].tolist())
    ind_58_rent_acc = accuracy_score(df_ind_58["true_label"], p58_rent)

    print(f"\nIndian Statements Evaluation:")
    print(f"  All 105 (MODEL-ONLY):     {ind_105_raw_acc*100:.2f}% [{i105_ci_low*100:.2f}%, {i105_ci_high*100:.2f}%] | ECE={ind_105_ece:.4f}")
    print(f"  All 105 (+Rent Rule):     {ind_105_rent_acc*100:.2f}% [{i105_rent_low*100:.2f}%, {i105_rent_high*100:.2f}%]")
    print(f"  Non-Ov 96 (MODEL-ONLY):   {ind_96_raw_acc*100:.2f}% [{i96_ci_low*100:.2f}%, {i96_ci_high*100:.2f}%]")
    print(f"  Non-Ov 96 (+Rent Rule):   {ind_96_rent_acc*100:.2f}%")
    print(f"  Disjoint 58 (MODEL-ONLY): {ind_58_raw_acc*100:.2f}% [{i58_ci_low*100:.2f}%, {i58_ci_high*100:.2f}%]")
    print(f"  Disjoint 58 (+Rent Rule): {ind_58_rent_acc*100:.2f}%")

    # 5. McNemar Significance Tests
    print("\n[Step 5] Running McNemar Significance Tests...")
    mcnemar_dict = {}

    # Load baseline predictions for comparison
    # Linear SVM (20k)
    pred_svm_20k = TransactionPredictor(model_path=MODELS_DIR / "variant_b_candidate.joblib")
    svm_20k_clean_preds = pred_svm_20k.predict_batch(test_split[["transaction_description"]])["predicted_category"].tolist()
    svm_20k_nv2_preds = pred_svm_20k.predict_batch(df_nv2_gt[["transaction_description"]])["predicted_category"].tolist()

    # Matched Linear SVM (3k)
    pred_svm_3k = TransactionPredictor(model_path=MODELS_DIR / "matched_linear_svm_3k.joblib")
    svm_3k_clean_preds = pred_svm_3k.predict_batch(test_split[["transaction_description"]])["predicted_category"].tolist()
    svm_3k_nv2_preds = pred_svm_3k.predict_batch(df_nv2_gt[["transaction_description"]])["predicted_category"].tolist()

    # Clean Test McNemar
    mc_clean_vs_svm20k = mcnemar_test(df_clean["true_label"], df_clean["pred_label"], svm_20k_clean_preds)
    mc_clean_vs_svm3k = mcnemar_test(df_clean["true_label"], df_clean["pred_label"], svm_3k_clean_preds)
    mcnemar_dict["DistilBERT_vs_Linear_SVM_20k_CleanTest"] = mc_clean_vs_svm20k
    mcnemar_dict["DistilBERT_vs_Matched_Linear_SVM_3k_CleanTest"] = mc_clean_vs_svm3k
    print(f"  DistilBERT vs Linear SVM (20k) [Clean]: chi2={mc_clean_vs_svm20k['statistic']}, p={mc_clean_vs_svm20k['p_value']}")
    print(f"  DistilBERT vs Matched SVM (3k) [Clean]: chi2={mc_clean_vs_svm3k['statistic']}, p={mc_clean_vs_svm3k['p_value']}")

    # Noisy V2 McNemar
    mc_nv2_vs_svm20k = mcnemar_test(df_noisy["true_label"], df_noisy["pred_label"], svm_20k_nv2_preds)
    mc_nv2_vs_svm3k = mcnemar_test(df_noisy["true_label"], df_noisy["pred_label"], svm_3k_nv2_preds)
    mcnemar_dict["DistilBERT_vs_Linear_SVM_20k_NoisyV2"] = mc_nv2_vs_svm20k
    mcnemar_dict["DistilBERT_vs_Matched_Linear_SVM_3k_NoisyV2"] = mc_nv2_vs_svm3k
    print(f"  DistilBERT vs Linear SVM (20k) [Noisy]: chi2={mc_nv2_vs_svm20k['statistic']}, p={mc_nv2_vs_svm20k['p_value']}")
    print(f"  DistilBERT vs Matched SVM (3k) [Noisy]: chi2={mc_nv2_vs_svm3k['statistic']}, p={mc_nv2_vs_svm3k['p_value']}")

    # Indian N=96 McNemar vs MiniLM and LightGBM
    pred_minilm = TransactionPredictor(model_path=MODELS_DIR / "minilm_lr_transaction_classifier.joblib")
    pred_lgb = TransactionPredictor(model_path=MODELS_DIR / "lightgbm_transaction_classifier.joblib")

    minilm_96_preds = pred_minilm.predict_batch(df_ind_96[["raw_text"]].rename(columns={"raw_text": "transaction_description"}))["predicted_category"].tolist()
    lgb_96_preds = pred_lgb.predict_batch(df_ind_96[["raw_text"]].rename(columns={"raw_text": "transaction_description"}))["predicted_category"].tolist()

    mc_ind_vs_minilm = mcnemar_test(df_ind_96["true_label"], df_ind_96["pred_label"], minilm_96_preds)
    mc_ind_vs_lgb = mcnemar_test(df_ind_96["true_label"], df_ind_96["pred_label"], lgb_96_preds)
    mcnemar_dict["DistilBERT_vs_MiniLM_Indian96_ModelOnly"] = mc_ind_vs_minilm
    mcnemar_dict["DistilBERT_vs_LightGBM_Indian96_ModelOnly"] = mc_ind_vs_lgb
    print(f"  DistilBERT vs MiniLM [Indian 96]:  chi2={mc_ind_vs_minilm['statistic']}, p={mc_ind_vs_minilm['p_value']}")
    print(f"  DistilBERT vs LightGBM [Indian 96]: chi2={mc_ind_vs_lgb['statistic']}, p={mc_ind_vs_lgb['p_value']}")

    with open(RESULTS_DIR / "mcnemar_tests_distilbert.json", "w") as f:
        json.dump(mcnemar_dict, f, indent=2)
    print(f"[SAVED] McNemar test results saved to {RESULTS_DIR / 'mcnemar_tests_distilbert.json'}")

    # 6. Append to results/final_metrics.csv
    new_row = {
        "Model": "DistilBERT_FineTuned",
        "Family": "Fine-Tuned Transformer",
        "Status": "MEASURED",
        "Validation_Accuracy": None,
        "Validation_Macro_F1": None,
        "Clean_Test_Accuracy": round(clean_acc, 4),
        "Clean_Test_95_CI": f"[{c_ci_low*100:.2f}%, {c_ci_high*100:.2f}%]",
        "Clean_Test_Macro_F1": round(clean_f1, 4),
        "Clean_Test_Weighted_F1": None,
        "Noisy_Test_Accuracy": None,
        "Noisy_Test_95_CI": None,
        "Noisy_Test_Macro_F1": None,
        "Noisy_Test_Weighted_F1": None,
        "Typo_Accuracy": None,
        "Typo_95_CI": None,
        "Indian_Model_Acc": round(ind_105_raw_acc, 4),
        "Indian_Model_95_CI": f"[{i105_ci_low*100:.2f}%, {i105_ci_high*100:.2f}%]",
        "Indian_Model_Plus_Rent_Acc": round(ind_105_rent_acc, 4),
        "Indian_Model_Plus_Rent_95_CI": f"[{i105_rent_low*100:.2f}%, {i105_rent_high*100:.2f}%]",
        "Indian_Statement_Macro_F1": round(ind_105_raw_f1, 4),
        "ECE": round(clean_ece, 4),
        "Single_Latency_With_Prep_ms": None,
        "Single_Latency_No_Prep_ms": None,
        "Batch256_Latency_With_Prep_ms": None,
        "Batch256_Latency_No_Prep_ms": None,
        "Training_Time_Sec": 348.5,
        "Artifact_Size_MB": None,
        "Noisy_V2_Accuracy": round(noisy_acc, 4),
        "Noisy_V2_95_CI": f"[{n_ci_low*100:.2f}%, {n_ci_high*100:.2f}%]",
        "Noisy_V2_Macro_F1": round(noisy_f1, 4)
    }

    df_final = pd.read_csv(FINAL_METRICS_PATH)
    # Check if DistilBERT_FineTuned already in final_metrics
    if not (df_final["Model"] == "DistilBERT_FineTuned").any():
        df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
        df_final.to_csv(FINAL_METRICS_PATH, index=False)
        print(f"[SAVED] Appended DistilBERT_FineTuned row to {FINAL_METRICS_PATH.name}")

    # Summary table output
    df_summary = pd.DataFrame([{
        "Model": "DistilBERT-base-uncased",
        "Clean_Test_Acc": round(clean_acc, 4),
        "Clean_Macro_F1": round(clean_f1, 4),
        "Clean_ECE": round(clean_ece, 4),
        "Noisy_V2_Acc": round(noisy_acc, 4),
        "Noisy_V2_Macro_F1": round(noisy_f1, 4),
        "Noisy_V2_ECE": round(noisy_ece, 4),
        "Indian_105_ModelOnly": round(ind_105_raw_acc, 4),
        "Indian_96_ModelOnly": round(ind_96_raw_acc, 4),
        "Indian_58_ModelOnly": round(ind_58_raw_acc, 4),
        "Indian_105_PlusRent": round(ind_105_rent_acc, 4),
        "Indian_96_PlusRent": round(ind_96_rent_acc, 4),
        "Indian_58_PlusRent": round(ind_58_rent_acc, 4),
        "Training_Time_Sec": 348.5,
        "Hardware": "2x T4 GPUs (Kaggle)"
    }])
    df_summary.to_csv(RESULTS_DIR / "distilbert_evaluation_summary.csv", index=False)
    pd.DataFrame(noise_breakdown).to_csv(RESULTS_DIR / "distilbert_noise_breakdown.csv", index=False)
    print(f"[SAVED] Summary saved to results/distilbert_evaluation_summary.csv and distilbert_noise_breakdown.csv")

if __name__ == "__main__":
    main()
