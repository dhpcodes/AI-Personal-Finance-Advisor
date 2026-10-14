"""compute_n96_stats.py — Wilson CIs and McNemar tests for the Indian N=96 subset.

What this script does
---------------------
1. Loads data/own_test_indian.csv (N=105 ground-truth rows).
2. Defines N=96 as rows whose canonicalized description does NOT match any of the
   8 overlapping merchant cores (amazon pay, dominos pizza, flipkart internet,
   myntra designs, ola cabs, swiggy, uber india, zomato), and N=58 as rows
   not matched by any core in the leakage-free seed — exactly as done in
   scripts/evaluate_distilbert_results.py.
3. Loads saved DistilBERT predictions from results/distilbert_preds_indian_105.csv.
4. Runs live inference using TransactionPredictor on:
     - models/variant_b_candidate.joblib  (Candidate Linear SVM, 20k features)
     - models/lightgbm_transaction_classifier.joblib
     - models/minilm_lr_transaction_classifier.joblib
   No models are retrained. No numbers are estimated.
5. Computes Wilson 95% CIs for model-only accuracy on N=105, N=96, N=58.
6. Computes McNemar exact tests (with Edwards continuity correction) on N=96 for:
     DistilBERT vs Candidate SVM, DistilBERT vs LightGBM, DistilBERT vs MiniLM,
     SVM vs LightGBM, SVM vs MiniLM.
7. Saves results/n96_stats.csv with all CIs and test results.
8. Also saves results/n96_per_row_predictions.csv with the per-row predictions
   from all four models on the 105 rows, for full auditability.

Run with:
    python scripts/compute_n96_stats.py
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score

from src.canonicalize import canonicalize
from src.predict import TransactionPredictor

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"
DISTILBERT_PREDS_PATH = RESULTS_DIR / "distilbert_preds_indian_105.csv"
CLEAN_SEED_PATH = DATA_DIR / "indian_merchants_train_seed_leakage_free.csv"

# Same overlapping cores as in scripts/evaluate_distilbert_results.py lines 49-52
OVERLAPPING_CORES = [
    "amazon pay", "dominos pizza", "flipkart internet", "myntra designs",
    "ola cabs", "swiggy", "uber india", "zomato"
]

MODEL_CONFIGS = [
    ("Candidate_Linear_SVM_20k", MODELS_DIR / "variant_b_candidate.joblib"),
    ("LightGBM", MODELS_DIR / "lightgbm_transaction_classifier.joblib"),
    ("MiniLM_LogReg", MODELS_DIR / "minilm_lr_transaction_classifier.joblib"),
]


def wilson_ci(acc: float, n: int, z: float = 1.95996):
    """Wilson score interval with continuity correction (95% CI)."""
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = acc + z**2 / (2 * n)
    margin = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact two-sided McNemar test (binomial, no continuity correction).

    b = n10 = Model A correct, Model B wrong.
    c = n01 = Model B correct, Model A wrong.
    p-value = 2 * P(X >= max(b,c)) under Binom(b+c, 0.5), capped at 1.
    Also reports chi2 with Edwards continuity correction for audit trail.
    """
    a_ok = np.array([a == y for a, y in zip(pred_a, y_true)])
    b_ok = np.array([b == y for b, y in zip(pred_b, y_true)])
    b_count = int(np.sum(a_ok & ~b_ok))   # n10
    c_count = int(np.sum(~a_ok & b_ok))   # n01
    n_disc = b_count + c_count
    if n_disc == 0:
        return {"b_n10": 0, "c_n01": 0, "n_discordant": 0,
                "exact_p": 1.0, "chi2_edwards": 0.0,
                "significant_p05": False, "significant_bonf_p01": False}
    from scipy.stats import binom as binom_dist, chi2 as chi2_dist
    m = max(b_count, c_count)
    exact_p = float(min(1.0, 2 * sum(binom_dist.pmf(k, n_disc, 0.5) for k in range(m, n_disc + 1))))
    stat = ((abs(b_count - c_count) - 1.0) ** 2) / n_disc
    p_chi2 = float(chi2_dist.sf(stat, df=1))
    return {
        "b_n10": b_count, "c_n01": c_count, "n_discordant": n_disc,
        "exact_p": round(exact_p, 5),
        "chi2_edwards": round(stat, 4),
        "p_chi2_edwards": round(p_chi2, 5),
        "significant_p05": exact_p < 0.05,
        "significant_bonf_p01": exact_p < 0.01,
    }


def main():
    print("=" * 70)
    print("COMPUTE N96 WILSON CIs AND MCNEMAR TESTS")
    print("=" * 70)

    # --- 1. Load ground-truth and DistilBERT predictions ---
    df_gt = pd.read_csv(OWN_TEST_PATH)
    df_db = pd.read_csv(DISTILBERT_PREDS_PATH)

    assert len(df_gt) == 105, f"Expected 105 rows, got {len(df_gt)}"
    assert len(df_db) == 105, f"Expected 105 DistilBERT rows, got {len(df_db)}"
    # Align by row order (evaluate_distilbert_results.py asserts text alignment)
    y_true_all = df_gt["category"].tolist()
    db_preds_all = df_db["pred_label"].tolist()

    # --- 2. Define N=96 mask (exclude overlapping cores) ---
    canons = [canonicalize(t) for t in df_gt["transaction_description"]]
    def is_overlapping(canon):
        c = canon.lower()
        return any(core in c for core in OVERLAPPING_CORES)
    ov_mask = np.array([is_overlapping(c) for c in canons])
    mask_96 = ~ov_mask
    n_ov = int(ov_mask.sum())
    n96 = int(mask_96.sum())
    print(f"Overlapping rows excluded: {n_ov}  →  N=96 mask has {n96} rows")

    # --- 3. Define N=58 mask (exclude rows matching leakage-free seed cores) ---
    df_seed = pd.read_csv(CLEAN_SEED_PATH)
    clean_seed_cores = set(df_seed["core"].dropna().str.lower().unique())
    def in_clean_seed(canon):
        c = canon.lower()
        return any(sc in c or c in sc for sc in clean_seed_cores)
    seed_mask = np.array([in_clean_seed(c) for c in canons])
    mask_58 = ~seed_mask
    n58 = int(mask_58.sum())
    print(f"Seed-matched rows excluded: {int(seed_mask.sum())}  →  N=58 mask has {n58} rows")

    # --- 4. Live inference for SVM / LightGBM / MiniLM ---
    print("\nRunning live inference on 105 Indian test rows...")
    all_preds = {"DistilBERT": db_preds_all}
    for model_name, model_path in MODEL_CONFIGS:
        if not model_path.exists():
            print(f"  [MISSING] {model_path.name} not found — skipping {model_name}")
            continue
        print(f"  Loading {model_path.name}...")
        predictor = TransactionPredictor(model_path=model_path)
        batch_df = df_gt[["transaction_description"]].copy()
        result_df = predictor.predict_batch(batch_df)
        preds = result_df["predicted_category"].tolist()
        all_preds[model_name] = preds
        acc105 = accuracy_score(y_true_all, preds)
        print(f"  {model_name}: N=105 model-only acc = {acc105:.4f}")

    # --- 5. Save per-row predictions for auditability ---
    per_row = df_gt[["transaction_description", "category"]].copy()
    per_row["canon"] = canons
    per_row["in_overlapping_9"] = ov_mask
    per_row["in_seed_47"] = seed_mask
    for model_name, preds in all_preds.items():
        per_row[f"pred_{model_name}"] = preds
        per_row[f"correct_{model_name}"] = [p == y for p, y in zip(preds, y_true_all)]
    per_row_path = RESULTS_DIR / "n96_per_row_predictions.csv"
    per_row.to_csv(per_row_path, index=False)
    print(f"\nSaved per-row predictions → {per_row_path.name}")

    # --- 6. Compute CIs for all N sizes ---
    rows = []
    model_names = list(all_preds.keys())
    masks = [("N105", np.ones(105, dtype=bool)), ("N96", mask_96), ("N58", mask_58)]

    for model_name in model_names:
        preds = all_preds[model_name]
        for subset_label, mask in masks:
            n = int(mask.sum())
            y_sub = [y for y, m in zip(y_true_all, mask) if m]
            p_sub = [p for p, m in zip(preds, mask) if m]
            acc = accuracy_score(y_sub, p_sub)
            lo, hi = wilson_ci(acc, n)
            rows.append({
                "Model": model_name,
                "Subset": subset_label,
                "N": n,
                "ModelOnly_Acc": round(acc, 4),
                "Wilson_95_CI_Lower": round(lo, 4),
                "Wilson_95_CI_Upper": round(hi, 4),
                "Wilson_95_CI_Formatted": f"[{lo*100:.2f}%, {hi*100:.2f}%]",
            })

    ci_df = pd.DataFrame(rows)

    # --- 7. McNemar tests on N=96 ---
    mcnemar_rows = []
    y96 = [y for y, m in zip(y_true_all, mask_96) if m]

    def get_preds_96(model_name):
        return [p for p, m in zip(all_preds[model_name], mask_96) if m]

    model_pairs = [
        ("DistilBERT", "Candidate_Linear_SVM_20k"),
        ("DistilBERT", "LightGBM"),
        ("DistilBERT", "MiniLM_LogReg"),
        ("Candidate_Linear_SVM_20k", "LightGBM"),
        ("Candidate_Linear_SVM_20k", "MiniLM_LogReg"),
    ]

    for m_a, m_b in model_pairs:
        if m_a not in all_preds or m_b not in all_preds:
            print(f"  [SKIP] Missing predictions for {m_a} or {m_b}")
            continue
        result = mcnemar_exact(y96, get_preds_96(m_a), get_preds_96(m_b))
        result.update({"Model_A": m_a, "Model_B": m_b, "Subset": "N96"})
        mcnemar_rows.append(result)
        print(f"  McNemar {m_a} vs {m_b}: b={result['b_n10']},c={result['c_n01']}, exact_p={result['exact_p']}, sig={result['significant_p05']}")

    mc_df = pd.DataFrame(mcnemar_rows)

    # --- 8. Save to results/n96_stats.csv ---
    out_path = RESULTS_DIR / "n96_stats.csv"
    # Merge CI and McNemar into one file with two sections via blank separator
    with open(out_path, "w", newline="") as fh:
        fh.write("# Section 1: Wilson 95% CIs\n")
        ci_df.to_csv(fh, index=False)
        fh.write("\n# Section 2: McNemar tests on N=96\n")
        mc_df[["Model_A", "Model_B", "Subset", "b_n10", "c_n01", "n_discordant",
               "exact_p", "chi2_edwards", "p_chi2_edwards",
               "significant_p05", "significant_bonf_p01"]].to_csv(fh, index=False)

    print(f"\nSaved → {out_path.name}")
    print("\n=== Wilson 95% CIs ===")
    print(ci_df.to_string(index=False))
    print("\n=== McNemar Tests on N=96 ===")
    print(mc_df[["Model_A", "Model_B", "b_n10", "c_n01", "exact_p", "significant_p05", "significant_bonf_p01"]].to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
