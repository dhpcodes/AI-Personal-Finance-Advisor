"""Independent Noise Test V2 Generator and Evaluator.

Strict Requirements:
1. DOES NOT import or reuse apply_single_noise().
2. Completely independent noise types:
   - vowel dropping
   - random char swaps
   - bank-specific suffixes
   - truncation at random positions
   - mixed corruption
3. Uses a completely independent seed (seed=8888).
4. Applied ONLY to the merchant-disjoint test split (N=20,040).
5. Saves corrupted dataset as data/noisy_test_v2.csv.
6. Evaluates all models:
   - Production Baseline (models/final_model.joblib)
   - Candidate Linear SVM (models/variant_b_candidate.joblib)
   - LightGBM (models/lightgbm_transaction_classifier.joblib)
   - XGBoost (models/xgboost_transaction_classifier.joblib)
   - MiniLM+LR (models/minilm_lr_transaction_classifier.joblib)
7. Computes 95% Wilson Score CIs and Macro-F1.
8. Updates results/final_metrics.csv and saves results/noise_test_v2_breakdown.csv.
"""

import sys
import os
import re
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.predict import TransactionPredictor

def wilson_score_interval(acc: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    z = 1.95996  # 95% CI
    denom = 1 + z**2 / n
    centre_adj = acc + z**2 / (2 * n)
    adj_val = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre_adj - adj_val) / denom)
    upper = min(1.0, (centre_adj + adj_val) / denom)
    return lower, upper

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
OUTPUT_NOISY_V2_PATH = DATA_DIR / "noisy_test_v2.csv"
FINAL_METRICS_PATH = RESULTS_DIR / "final_metrics.csv"

BANK_SUFFIXES = [
    "/CMS/REF88129",
    "/BIL/HDFC-AUTO",
    " CR/UTR-92812039",
    "/SETTLE-ICICI/01",
    "-TXN-NETBANKING-991",
    "/ACH-DEBIT-901",
    "/NEFT-INW-00912",
    "/NACH/MANDATE-552",
    " /RTGS-SETTL-8821",
    "/INB/TXN882012"
]

def apply_vowel_drop(text: str, rng: random.Random) -> str:
    """Drop vowels from words of length > 3 with probability 0.7."""
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 3:
            # Keep first letter, remove vowels from rest
            first, rest = w[0], w[1:]
            no_vowels = re.sub(r"[aeiouAEIOU]", "", rest)
            new_words.append(first + no_vowels if no_vowels else w)
        else:
            new_words.append(w)
    return " ".join(new_words)

def apply_char_swap(text: str, rng: random.Random) -> str:
    """Swap two adjacent characters within a word."""
    s = list(text)
    if len(s) >= 4:
        idx = rng.randint(1, len(s) - 2)
        s[idx], s[idx + 1] = s[idx + 1], s[idx]
    return "".join(s)

def apply_bank_suffix(text: str, rng: random.Random) -> str:
    """Append a realistic banking settlement / clearing suffix."""
    suffix = rng.choice(BANK_SUFFIXES)
    return f"{text}{suffix}"

def apply_random_truncation(text: str, rng: random.Random) -> str:
    """Truncate text at a random position between 40% and 75% length."""
    if len(text) <= 5:
        return text
    keep_len = rng.randint(max(3, int(len(text) * 0.40)), max(4, int(len(text) * 0.75)))
    return text[:keep_len]

def apply_mixed_noise(text: str, rng: random.Random) -> str:
    """Combine vowel dropping or char swap with a bank suffix."""
    if rng.random() < 0.5:
        t = apply_vowel_drop(text, rng)
    else:
        t = apply_char_swap(text, rng)
    return apply_bank_suffix(t, rng)

def corrupt_independent(text: str, rng: random.Random) -> tuple[str, str]:
    """Select one independent noise type at random."""
    choice = rng.choice(["vowel_drop", "char_swap", "bank_suffix", "random_truncation", "mixed_noise"])
    if choice == "vowel_drop":
        return apply_vowel_drop(text, rng), "vowel_drop"
    elif choice == "char_swap":
        return apply_char_swap(text, rng), "char_swap"
    elif choice == "bank_suffix":
        return apply_bank_suffix(text, rng), "bank_suffix"
    elif choice == "random_truncation":
        return apply_random_truncation(text, rng), "random_truncation"
    else:
        return apply_mixed_noise(text, rng), "mixed_noise"

def main():
    print("=" * 70)
    print("GENERATING INDEPENDENT NOISY TEST V2 & EVALUATING ALL MODELS")
    print("=" * 70)

    # 1. Load test split
    print(f"\n[Step 1] Loading merchant-disjoint test split from {SPLIT_PATH}...")
    df_split = pd.read_csv(SPLIT_PATH)
    test_df = df_split[df_split.split == "test"].copy()
    print(f"Loaded {len(test_df):,} test records.")

    # 2. Corrupt with independent seed
    SEED = 8888
    rng = random.Random(SEED)
    print(f"\n[Step 2] Corrupting descriptions with independent seed {SEED}...")

    noisy_descs = []
    noise_types = []
    for raw_desc in test_df["transaction_description"]:
        corrupted, n_type = corrupt_independent(str(raw_desc), rng)
        noisy_descs.append(corrupted)
        noise_types.append(n_type)

    df_noisy_v2 = pd.DataFrame({
        "transaction_description": noisy_descs,
        "original_description": test_df["transaction_description"].tolist(),
        "category": test_df["category"].tolist(),
        "noise_type": noise_types,
        "split": "test"
    })

    df_noisy_v2.to_csv(OUTPUT_NOISY_V2_PATH, index=False)
    print(f"[SAVED] Saved {len(df_noisy_v2):,} noisy V2 test records to {OUTPUT_NOISY_V2_PATH}")
    print("\nNoise type distribution:")
    print(df_noisy_v2["noise_type"].value_counts())

    print("\nSample corruptions:")
    for i in range(5):
        row = df_noisy_v2.iloc[i]
        print(f"  [{row['noise_type']}]")
        print(f"    Orig:  {row['original_description']}")
        print(f"    Noisy: {row['transaction_description']}")

    # 3. Evaluate all models
    models = {
        "Production_Baseline": MODELS_DIR / "final_model.joblib",
        "Candidate_Linear_SVM": MODELS_DIR / "variant_b_candidate.joblib",
        "LightGBM": MODELS_DIR / "lightgbm_transaction_classifier.joblib",
        "XGBoost": MODELS_DIR / "xgboost_transaction_classifier.joblib",
        "MiniLM_LogReg": MODELS_DIR / "minilm_lr_transaction_classifier.joblib",
    }

    y_true = df_noisy_v2["category"].tolist()
    eval_results = []
    breakdown_results = []

    print("\n[Step 3] Evaluating models on Noisy Test V2...")

    for name, model_path in models.items():
        print(f"\n--- Evaluating {name} ---")
        if not model_path.exists():
            print(f"Model path {model_path} not found!")
            continue

        predictor = TransactionPredictor(model_path=model_path)
        t0 = time.time()
        res = predictor.predict_batch(df_noisy_v2[["transaction_description"]])
        preds = res["predicted_category"].tolist()
        dur = time.time() - t0

        acc = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average="macro")
        ci_low, ci_high = wilson_score_interval(acc, len(y_true))
        ci_str = f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]"

        print(f"  Overall Noisy V2 Accuracy: {acc:.4f} ({acc*100:.2f}%) {ci_str}")
        print(f"  Overall Noisy V2 Macro-F1: {macro_f1:.4f}")
        print(f"  Inference Time: {dur:.2f}s ({len(y_true)/max(dur,0.001):.1f} txn/s)")

        eval_results.append({
            "Model": name,
            "Noisy_V2_Accuracy": acc,
            "Noisy_V2_95_CI": ci_str,
            "Noisy_V2_Macro_F1": macro_f1
        })

        # Breakdown by noise type
        for n_type in df_noisy_v2["noise_type"].unique():
            mask = (df_noisy_v2["noise_type"] == n_type).values
            sub_y = [y_true[i] for i in range(len(y_true)) if mask[i]]
            sub_preds = [preds[i] for i in range(len(preds)) if mask[i]]
            sub_acc = accuracy_score(sub_y, sub_preds)
            breakdown_results.append({
                "Model": name,
                "Noise_Type": n_type,
                "Samples": len(sub_y),
                "Accuracy": sub_acc
            })

    df_eval = pd.DataFrame(eval_results)
    df_breakdown = pd.DataFrame(breakdown_results)

    breakdown_path = RESULTS_DIR / "noise_test_v2_breakdown.csv"
    df_breakdown.to_csv(breakdown_path, index=False)
    print(f"\n[SAVED] Noise type breakdown saved to {breakdown_path}")

    # 4. Update results/final_metrics.csv
    if FINAL_METRICS_PATH.exists():
        df_final = pd.read_csv(FINAL_METRICS_PATH)
        # Merge Noisy_V2 columns
        merged = df_final.merge(df_eval, on="Model", how="left")
        merged.to_csv(FINAL_METRICS_PATH, index=False)
        print(f"[SAVED] Updated {FINAL_METRICS_PATH} with Noisy_V2 metrics.")

    print("\n" + "=" * 70)
    print("NOISY TEST V2 EVALUATION SUMMARY:")
    print("=" * 70)
    print(df_eval.to_string(index=False))

    print("\nPER-NOISE-TYPE ACCURACY SUMMARY:")
    pivot_breakdown = df_breakdown.pivot(index="Model", columns="Noise_Type", values="Accuracy")
    print(pivot_breakdown.map(lambda x: f"{x*100:.2f}%").to_string())

if __name__ == "__main__":
    main()
