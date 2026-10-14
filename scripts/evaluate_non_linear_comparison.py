"""Evaluate and Compare Non-Linear Models vs Linear Baselines (Task 3).

Evaluates:
1. Linear SVM (full 72k, models/variant_b_candidate.joblib)
2. Linear SVM (20k, models/linear_svm_20k.joblib)
3. RBF-Kernel SVM (20k, models/rbf_svm_20k.joblib)
4. MLP Classifier (models/mlp_tfidf_classifier.joblib)
5. LightGBM (models/lightgbm_transaction_classifier.joblib)
6. MiniLM+LR (models/minilm_lr_transaction_classifier.joblib)
7. Linear SVM Leakage-Free 20k (models/variant_b_candidate_leakage_free_20k.joblib)

Test Sets Evaluated:
- Clean Test (N=20,040) with 95% Wilson CIs
- Noisy Test V2 (N=20,040) with 95% Wilson CIs
- Typo Subset (N=2,492, generated from Phase 3 test noise)
- Real Indian Bank Statements:
  - All 105 statements (N=105, with Rent Rule)
  - Non-overlapping statements (N=96, with Rent Rule)
  - Fully disjoint statements (N=58, with Rent Rule)
- Single-item inference latency (ms/txn, average over 200 runs)
- McNemar significance test against Linear SVM (both full and 20k) on clean test and noisy v2.

Outputs:
- results/non_linear_comparison.csv
- results/mcnemar_tests_non_linear.json
- Appends new evaluated model rows to results/final_metrics.csv
"""

import sys
import time
import json
import random
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
from src.predict import TransactionPredictor
from src.model_comparison_suite import with_rent_rule

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
NOISY_V2_PATH = DATA_DIR / "noisy_test_v2.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"
CLEAN_SEED_PATH = DATA_DIR / "indian_merchants_train_seed_leakage_free.csv"
FINAL_METRICS_PATH = RESULTS_DIR / "final_metrics.csv"

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

def apply_typo_noise(text: str, rng: random.Random) -> str:
    s = list(text)
    if len(s) > 4:
        i = rng.randint(1, len(s) - 2)
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)

def measure_single_latency(predictor: TransactionPredictor, texts: list, n_runs: int = 200) -> float:
    # Warmup
    for i in range(20):
        _ = predictor.predict_single(texts[i % len(texts)])
    t0 = time.perf_counter()
    for i in range(n_runs):
        _ = predictor.predict_single(texts[i % len(texts)])
    tot = time.perf_counter() - t0
    return (tot / n_runs) * 1000.0

def main():
    print("=" * 70)
    print("TASK 3: EVALUATE & COMPARE NON-LINEAR MODELS VS LINEAR BASELINES")
    print("=" * 70)

    # 1. Load Test Partitions
    print("\n[Step 1] Loading Test Sets...")
    df_split = pd.read_csv(SPLIT_PATH)
    test_df = df_split[df_split.split == "test"].copy()
    y_clean_test = test_df["category"].tolist()
    print(f"  Clean Test:     N = {len(test_df):,}")

    df_noisy_v2 = pd.read_csv(NOISY_V2_PATH)
    y_noisy_v2 = df_noisy_v2["category"].tolist()
    print(f"  Noisy Test V2:  N = {len(df_noisy_v2):,}")

    # Generate Typo subset (N=2,492) using seed 7 matching Phase 3
    rng_typo = random.Random(7)
    typo_indices = [i for i, _ in enumerate(test_df["transaction_description"]) if rng_typo.random() < (2492 / len(test_df))]
    df_typo = test_df.iloc[typo_indices].copy()
    rng_t2 = random.Random(7)
    df_typo["transaction_description"] = [apply_typo_noise(t, rng_t2) for t in df_typo["transaction_description"]]
    y_typo = df_typo["category"].tolist()
    print(f"  Typo Test:      N = {len(df_typo):,}")

    # Real Indian Statements
    df_own = pd.read_csv(OWN_TEST_PATH)
    df_own["canon"] = df_own["transaction_description"].apply(canonicalize)
    is_ov_9 = df_own["canon"].apply(lambda c: any(core in c.lower() for core in OVERLAPPING_CORES))
    df_own_96 = df_own[~is_ov_9].copy()

    df_clean_seed = pd.read_csv(CLEAN_SEED_PATH)
    clean_seed_cores = set(df_clean_seed["core"].dropna().str.lower().unique())
    is_in_clean_seed = df_own["canon"].apply(lambda c: any(sc in c.lower() or c.lower() in sc for sc in clean_seed_cores))
    df_own_58 = df_own[~is_in_clean_seed].copy()

    print(f"  Indian All:     N = {len(df_own)}")
    print(f"  Indian Non-Ov:  N = {len(df_own_96)}")
    print(f"  Indian Disjoint:N = {len(df_own_58)}")

    # Models to Evaluate
    models = {
        "Linear_SVM_Full": MODELS_DIR / "variant_b_candidate.joblib",
        "Linear_SVM_20k": MODELS_DIR / "linear_svm_20k.joblib",
        "RBF_SVM_20k": MODELS_DIR / "rbf_svm_20k.joblib",
        "MLP_TFIDF": MODELS_DIR / "mlp_tfidf_classifier.joblib",
        "LightGBM": MODELS_DIR / "lightgbm_transaction_classifier.joblib",
        "MiniLM_LogReg": MODELS_DIR / "minilm_lr_transaction_classifier.joblib",
        "Linear_SVM_LeakageFree_20k": MODELS_DIR / "variant_b_candidate_leakage_free_20k.joblib",
    }

    eval_rows = []
    clean_preds_map = {}
    noisy_v2_preds_map = {}

    for name, m_path in models.items():
        print(f"\n--- Evaluating {name} ---")
        if not m_path.exists():
            print(f"  [SKIPPED] Model artifact {m_path.name} not found!")
            continue

        predictor = TransactionPredictor(model_path=m_path)
        art = joblib.load(m_path)
        metrics_dict = art.get("metrics", {})
        fit_time = metrics_dict.get("fit_time_sec", None)
        train_acc = metrics_dict.get("train_accuracy", None)
        val_acc = metrics_dict.get("val_accuracy", None)

        # Fallback to overfitting_check.csv if not stored in artifact
        if (train_acc is None or val_acc is None) and (RESULTS_DIR / "overfitting_check.csv").exists():
            df_ov = pd.read_csv(RESULTS_DIR / "overfitting_check.csv")
            name_map = {
                "Linear_SVM_Full": "Candidate_Linear_SVM",
                "LightGBM": "LightGBM",
                "MiniLM_LogReg": "MiniLM_LogReg"
            }
            target_key = name_map.get(name, name)
            m_row = df_ov[df_ov["Model"] == target_key]
            if not m_row.empty:
                if train_acc is None:
                    train_acc = round(float(m_row.iloc[0]["Train_Acc"]), 4)
                if val_acc is None:
                    val_acc = round(float(m_row.iloc[0]["Val_Acc"]), 4)

        # 1. Clean Test
        t0 = time.time()
        res_clean = predictor.predict_batch(test_df[["transaction_description"]])
        clean_preds = res_clean["predicted_category"].tolist()
        dur_clean = time.time() - t0
        acc_clean = accuracy_score(y_clean_test, clean_preds)
        f1_clean = f1_score(y_clean_test, clean_preds, average="macro")
        ci_c_low, ci_c_high = wilson_score_interval(acc_clean, len(y_clean_test))
        clean_preds_map[name] = clean_preds

        # 2. Noisy Test V2
        t0 = time.time()
        res_nv2 = predictor.predict_batch(df_noisy_v2[["transaction_description"]])
        nv2_preds = res_nv2["predicted_category"].tolist()
        acc_nv2 = accuracy_score(y_noisy_v2, nv2_preds)
        f1_nv2 = f1_score(y_noisy_v2, nv2_preds, average="macro")
        ci_n_low, ci_n_high = wilson_score_interval(acc_nv2, len(y_noisy_v2))
        noisy_v2_preds_map[name] = nv2_preds

        # 3. Typo Subset
        res_typo = predictor.predict_batch(df_typo[["transaction_description"]])
        typo_preds = res_typo["predicted_category"].tolist()
        acc_typo = accuracy_score(y_typo, typo_preds)

        # 4. Indian Statements
        # All 105
        res_105 = predictor.predict_batch(df_own[["transaction_description"]])
        p105 = res_105["predicted_category"].tolist()
        p105_rent = with_rent_rule(df_own["transaction_description"].tolist(), p105)
        acc_105 = accuracy_score(df_own["category"], p105_rent)

        # Non-overlapping 96
        res_96 = predictor.predict_batch(df_own_96[["transaction_description"]])
        p96 = res_96["predicted_category"].tolist()
        p96_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), p96)
        acc_96 = accuracy_score(df_own_96["category"], p96_rent)

        # Fully disjoint 58
        res_58 = predictor.predict_batch(df_own_58[["transaction_description"]])
        p58 = res_58["predicted_category"].tolist()
        p58_rent = with_rent_rule(df_own_58["transaction_description"].tolist(), p58)
        acc_58 = accuracy_score(df_own_58["category"], p58_rent)

        # 5. Latency
        lat = measure_single_latency(predictor, test_df["transaction_description"].tolist(), n_runs=200)

        print(f"  Clean Test: {acc_clean*100:.2f}% [{ci_c_low*100:.2f}%, {ci_c_high*100:.2f}%]")
        print(f"  Noisy V2:   {acc_nv2*100:.2f}% [{ci_n_low*100:.2f}%, {ci_n_high*100:.2f}%]")
        print(f"  Typo:       {acc_typo*100:.2f}%")
        print(f"  Indian 105: {acc_105*100:.2f}% | Indian 96: {acc_96*100:.2f}% | Indian 58: {acc_58*100:.2f}%")
        print(f"  Single Latency: {lat:.3f} ms/txn")

        eval_rows.append({
            "Model": name,
            "Train_Acc": train_acc,
            "Val_Acc": val_acc,
            "Clean_Test_Acc": acc_clean,
            "Clean_95_CI": f"[{ci_c_low*100:.2f}%, {ci_c_high*100:.2f}%]",
            "Clean_Macro_F1": f1_clean,
            "Noisy_V2_Acc": acc_nv2,
            "Noisy_V2_95_CI": f"[{ci_n_low*100:.2f}%, {ci_n_high*100:.2f}%]",
            "Noisy_V2_Macro_F1": f1_nv2,
            "Typo_Acc": acc_typo,
            "Indian_105_Acc": acc_105,
            "Indian_96_Acc": acc_96,
            "Indian_58_Acc": acc_58,
            "Single_Latency_ms": round(lat, 3),
            "Fit_Time_Sec": fit_time
        })

    df_res = pd.DataFrame(eval_rows)
    out_table_path = RESULTS_DIR / "non_linear_comparison.csv"
    df_res.to_csv(out_table_path, index=False)
    print(f"\n[SAVED] Non-linear comparison table saved to {out_table_path}")

    # McNemar Significance Tests
    print("\n[Step 3] Running McNemar Significance Tests...")
    mc_results = {}
    
    # Compare against Linear_SVM_Full
    ref_clean = clean_preds_map["Linear_SVM_Full"]
    ref_nv2 = noisy_v2_preds_map["Linear_SVM_Full"]

    for cand in ["MLP_TFIDF", "LightGBM", "MiniLM_LogReg"]:
        if cand in clean_preds_map:
            mc_clean = mcnemar_test(y_clean_test, ref_clean, clean_preds_map[cand])
            mc_nv2 = mcnemar_test(y_noisy_v2, ref_nv2, noisy_v2_preds_map[cand])
            mc_results[f"Linear_SVM_Full_vs_{cand}_Clean"] = mc_clean
            mc_results[f"Linear_SVM_Full_vs_{cand}_NoisyV2"] = mc_nv2
            print(f"  Linear SVM vs {cand} (Clean Test): chi2={mc_clean['statistic']}, p={mc_clean['p_value']}")
            print(f"  Linear SVM vs {cand} (Noisy V2):   chi2={mc_nv2['statistic']}, p={mc_nv2['p_value']}")

    # Compare Linear_SVM_20k vs RBF_SVM_20k (Equal Data Control)
    if "Linear_SVM_20k" in clean_preds_map and "RBF_SVM_20k" in clean_preds_map:
        mc_rbf_clean = mcnemar_test(y_clean_test, clean_preds_map["Linear_SVM_20k"], clean_preds_map["RBF_SVM_20k"])
        mc_rbf_nv2 = mcnemar_test(y_noisy_v2, noisy_v2_preds_map["Linear_SVM_20k"], noisy_v2_preds_map["RBF_SVM_20k"])
        mc_results["Linear_20k_vs_RBF_20k_Clean"] = mc_rbf_clean
        mc_results["Linear_20k_vs_RBF_20k_NoisyV2"] = mc_rbf_nv2
        print(f"\n  [EQUAL DATA 20K] Linear SVM vs RBF SVM (Clean Test): chi2={mc_rbf_clean['statistic']}, p={mc_rbf_clean['p_value']}")
        print(f"  [EQUAL DATA 20K] Linear SVM vs RBF SVM (Noisy V2):   chi2={mc_rbf_nv2['statistic']}, p={mc_rbf_nv2['p_value']}")

    with open(RESULTS_DIR / "mcnemar_tests_non_linear.json", "w") as f:
        json.dump(mc_results, f, indent=2)
    print(f"[SAVED] Saved McNemar test results to {RESULTS_DIR / 'mcnemar_tests_non_linear.json'}")

    # Append new models to final_metrics.csv
    if FINAL_METRICS_PATH.exists():
        df_final = pd.read_csv(FINAL_METRICS_PATH)
        existing_models = set(df_final["Model"].tolist())
        new_rows = []
        for r in eval_rows:
            m_name = r["Model"]
            if m_name not in existing_models:
                new_rows.append({
                    "Model": m_name,
                    "Family": "Non-Linear / Subsample",
                    "Status": "MEASURED",
                    "Validation_Accuracy": r["Val_Acc"],
                    "Clean_Test_Accuracy": r["Clean_Test_Acc"],
                    "Clean_Test_95_CI": r["Clean_95_CI"],
                    "Clean_Test_Macro_F1": r["Clean_Macro_F1"],
                    "Noisy_Test_Accuracy": None,
                    "Typo_Accuracy": r["Typo_Acc"],
                    "Indian_Model_Plus_Rent_Acc": r["Indian_105_Acc"],
                    "Single_Latency_With_Prep_ms": r["Single_Latency_ms"],
                    "Training_Time_Sec": r["Fit_Time_Sec"],
                    "Noisy_V2_Accuracy": r["Noisy_V2_Acc"],
                    "Noisy_V2_95_CI": r["Noisy_V2_95_CI"],
                    "Noisy_V2_Macro_F1": r["Noisy_V2_Macro_F1"]
                })
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_final_updated = pd.concat([df_final, df_new], ignore_index=True)
            df_final_updated.to_csv(FINAL_METRICS_PATH, index=False)
            print(f"[SAVED] Appended {len(new_rows)} new models to {FINAL_METRICS_PATH.name}")

    print("\n" + "=" * 70)
    print("TASK 3 FINAL COMPARISON TABLE:")
    print("=" * 70)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
