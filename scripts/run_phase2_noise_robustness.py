"""Phase 2: Noise Robustness Experiment Suite.

Computes:
2b. Ablation matrix (Raw vs Canonicalized, With vs Without noise augmentation).
2c. Per-noise-type evaluation (UPI, POS, NEFT, IMPS, upper, lower, trunc, typo).
2d. Temperature scaling calibration on noisy validation set.
2e. Label noise / out-of-fold disagreement analysis to establish accuracy ceiling.
"""

import random
import re
import sys
import time
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.canonicalize import canonicalize

SPLIT_PATH = Path("data/mitulshah_split.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]
TEST_KINDS  = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]


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


def ece_score(y_true: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            e += k.mean() * abs((pred[k] == y_true[k]).mean() - conf[k].mean())
    return float(e)


class TemperatureScaler:
    """Post-hoc temperature scaling fitted via NLL on validation set."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        # Minimize NLL w.r.t temperature T
        def nll(t):
            t_val = t[0]
            scaled = logits / max(t_val, 1e-3)
            # Softmax
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
            # Log loss
            y_one_hot = np.zeros_like(probs)
            y_one_hot[np.arange(len(y_true)), y_true] = 1.0
            return -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))

        res = minimize(nll, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


def main():
    print("=" * 70)
    print("PHASE 2: NOISE ROBUSTNESS BENCHMARKS & EXPERIMENTS")
    print("=" * 70)

    df = pd.read_csv(SPLIT_PATH)
    train_clean = df[df.split == "train"].copy()
    val_clean   = df[df.split == "val"].copy()
    test_clean  = df[df.split == "test"].copy()

    # Pre-generate noisy versions
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train_clean)) < 0.35
    train_noisy = train_clean.copy()
    train_noisy.loc[m_tr, "transaction_description"] = [
        apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) for t in train_noisy.loc[m_tr, "transaction_description"]
    ]

    r_va = random.Random(11)
    m_va = np.random.RandomState(1).rand(len(val_clean)) < 0.35
    val_noisy = val_clean.copy()
    val_noisy.loc[m_va, "transaction_description"] = [
        apply_single_noise(t, r_va, r_va.choice(TRAIN_KINDS)) for t in val_noisy.loc[m_va, "transaction_description"]
    ]

    # Pre-generate noisy test set with balanced noise type annotations for 2c
    test_noisy = test_clean.copy()
    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_noisy))]
    test_noisy["noise_type"] = test_noise_types
    test_noisy["transaction_description"] = [
        apply_single_noise(t, r_te, nt) for t, nt in zip(test_clean["transaction_description"], test_noise_types)
    ]

    le = LabelEncoder().fit(df.category)
    yte_clean = le.transform(test_clean.category)
    yte_noisy = le.transform(test_noisy.category)
    yva_noisy = le.transform(val_noisy.category)

    # -------------------------------------------------------------
    # 2b. ABLATION MATRIX (Raw vs Canonicalized, With vs Without Noise Aug)
    # -------------------------------------------------------------
    print("\n--- Running 2b: Noise Ablation Matrix across Seeds 42, 43, 44 ---")
    SEEDS = [42, 43, 44]
    ablation_records = []

    for use_canonical in [False, True]:
        canon_label = "Canonicalized" if use_canonical else "Raw Text"

        # Apply canonicalization if requested
        if use_canonical:
            tr_clean_texts = [canonicalize(t) for t in train_clean.transaction_description]
            tr_noisy_texts = [canonicalize(t) for t in train_noisy.transaction_description]
            va_noisy_texts = [canonicalize(t) for t in val_noisy.transaction_description]
            te_clean_texts = [canonicalize(t) for t in test_clean.transaction_description]
            te_noisy_texts = [canonicalize(t) for t in test_noisy.transaction_description]
        else:
            tr_clean_texts = train_clean.transaction_description.tolist()
            tr_noisy_texts = train_noisy.transaction_description.tolist()
            va_noisy_texts = val_noisy.transaction_description.tolist()
            te_clean_texts = test_clean.transaction_description.tolist()
            te_noisy_texts = test_noisy.transaction_description.tolist()

        for use_noise_aug in [False, True]:
            aug_label = "With Noise Aug (35%)" if use_noise_aug else "No Noise Aug (Clean Train)"
            tr_texts = tr_noisy_texts if use_noise_aug else tr_clean_texts

            tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=3,
                max_features=20000,
                sublinear_tf=True,
                lowercase=False
            )
            Xtr = tfidf.fit_transform(tr_texts)
            ytr = le.transform(train_clean.category)

            Xte_clean = tfidf.transform(te_clean_texts)
            Xte_noisy = tfidf.transform(te_noisy_texts)

            for seed in SEEDS:
                # 1. LogReg
                lr = LogisticRegression(C=10, max_iter=300, random_state=seed)
                lr.fit(Xtr, ytr)
                p_c = lr.predict(Xte_clean)
                p_n = lr.predict(Xte_noisy)

                ablation_records.append({
                    "Model": "Logistic Regression",
                    "Input_Format": canon_label,
                    "Training_Augmentation": aug_label,
                    "Seed": seed,
                    "Clean_Test_Acc": accuracy_score(yte_clean, p_c),
                    "Clean_Test_MacroF1": f1_score(yte_clean, p_c, average="macro"),
                    "Noisy_Test_Acc": accuracy_score(yte_noisy, p_n),
                    "Noisy_Test_MacroF1": f1_score(yte_noisy, p_n, average="macro")
                })

                # 2. Linear SVM
                svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, random_state=seed), cv=2)
                svm.fit(Xtr, ytr)
                p_c_svm = svm.predict(Xte_clean)
                p_n_svm = svm.predict(Xte_noisy)

                ablation_records.append({
                    "Model": "Linear SVM",
                    "Input_Format": canon_label,
                    "Training_Augmentation": aug_label,
                    "Seed": seed,
                    "Clean_Test_Acc": accuracy_score(yte_clean, p_c_svm),
                    "Clean_Test_MacroF1": f1_score(yte_clean, p_c_svm, average="macro"),
                    "Noisy_Test_Acc": accuracy_score(yte_noisy, p_n_svm),
                    "Noisy_Test_MacroF1": f1_score(yte_noisy, p_n_svm, average="macro")
                })

    abl_df = pd.DataFrame(ablation_records)
    summary_abl = abl_df.groupby(["Model", "Input_Format", "Training_Augmentation"]).agg({
        "Clean_Test_Acc": ["mean", "std"],
        "Clean_Test_MacroF1": ["mean", "std"],
        "Noisy_Test_Acc": ["mean", "std"],
        "Noisy_Test_MacroF1": ["mean", "std"]
    }).round(4)
    summary_abl.to_csv(RESULTS_DIR / "noise_ablation_matrix.csv")
    print("Saved results/noise_ablation_matrix.csv")
    print(summary_abl)

    # -------------------------------------------------------------
    # 2c. PER-NOISE-TYPE BREAKDOWN ON NOISY TEST SET
    # -------------------------------------------------------------
    print("\n--- Running 2c: Per-Noise-Type Evaluation Breakdown ---")
    # Using Seed 42 benchmark models (Raw vs Canonicalized with Noise Aug)
    # Train Raw model
    tfidf_raw = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
    Xtr_raw = tfidf_raw.fit_transform(train_noisy.transaction_description)
    lr_raw = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_raw, le.transform(train_noisy.category))

    # Train Canonicalized model
    tfidf_can = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
    tr_can = [canonicalize(t) for t in train_noisy.transaction_description]
    Xtr_can = tfidf_can.fit_transform(tr_can)
    lr_can = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_can, le.transform(train_noisy.category))

    noise_records = []
    for kind in TEST_KINDS:
        sub_mask = test_noisy["noise_type"] == kind
        sub_df = test_noisy[sub_mask]
        sub_y = yte_noisy[sub_mask]

        # Raw evaluation
        X_sub_raw = tfidf_raw.transform(sub_df.transaction_description)
        p_raw = lr_raw.predict(X_sub_raw)
        acc_raw = accuracy_score(sub_y, p_raw)
        f1_raw = f1_score(sub_y, p_raw, average="macro")

        # Canonicalized evaluation
        sub_can = [canonicalize(t) for t in sub_df.transaction_description]
        X_sub_can = tfidf_can.transform(sub_can)
        p_can = lr_can.predict(X_sub_can)
        acc_can = accuracy_score(sub_y, p_can)
        f1_can = f1_score(sub_y, p_can, average="macro")

        is_held_out = "HELD-OUT FROM TRAINING" if kind in ["neft", "imps"] else "Included in Train Noise"
        noise_records.append({
            "Noise_Type": kind.upper(),
            "Sample_Count": len(sub_df),
            "Training_Status": is_held_out,
            "Raw_LogReg_Acc": round(acc_raw, 4),
            "Raw_LogReg_MacroF1": round(f1_raw, 4),
            "Canonicalized_LogReg_Acc": round(acc_can, 4),
            "Canonicalized_LogReg_MacroF1": round(f1_can, 4),
            "Accuracy_Delta_Pct": round((acc_can - acc_raw) * 100, 2)
        })

    noise_df = pd.DataFrame(noise_records)
    noise_df.to_csv(RESULTS_DIR / "per_noise_type_evaluation.csv", index=False)
    print("Saved results/per_noise_type_evaluation.csv")
    print(noise_df[["Noise_Type", "Sample_Count", "Training_Status", "Raw_LogReg_Acc", "Canonicalized_LogReg_Acc", "Accuracy_Delta_Pct"]])

    # -------------------------------------------------------------
    # 2d. TEMPERATURE SCALING CALIBRATION
    # -------------------------------------------------------------
    print("\n--- Running 2d: Temperature Scaling Calibration ---")
    # Extract decision function (logits) on validation set ONLY
    val_can = [canonicalize(t) for t in val_noisy.transaction_description]
    Xva_can = tfidf_can.transform(val_can)
    val_logits = lr_can.decision_function(Xva_can)

    temp_scaler = TemperatureScaler().fit(val_logits, yva_noisy)
    T = temp_scaler.temperature
    print(f"Optimal Temperature parameter T fitted on Noisy Validation: {T:.4f}")

    # Evaluate on Clean Test and Noisy Test
    Xte_can_clean = tfidf_can.transform([canonicalize(t) for t in test_clean.transaction_description])
    Xte_can_noisy = tfidf_can.transform([canonicalize(t) for t in test_noisy.transaction_description])

    logits_clean = lr_can.decision_function(Xte_can_clean)
    logits_noisy = lr_can.decision_function(Xte_can_noisy)

    # Raw Softmax probabilities
    def to_probs(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    probs_clean_raw = to_probs(logits_clean)
    probs_noisy_raw = to_probs(logits_noisy)

    probs_clean_cal = temp_scaler.transform(logits_clean)
    probs_noisy_cal = temp_scaler.transform(logits_noisy)

    ece_clean_raw = ece_score(yte_clean, probs_clean_raw)
    ece_clean_cal = ece_score(yte_clean, probs_clean_cal)

    ece_noisy_raw = ece_score(yte_noisy, probs_noisy_raw)
    ece_noisy_cal = ece_score(yte_noisy, probs_noisy_cal)

    calib_records = [
        {
            "Dataset": "Clean Test (Unseen Merchants)",
            "Model": "Canonicalized LogReg",
            "Optimal_Temperature_T": round(T, 4),
            "ECE_Before_Calibration": round(ece_clean_raw, 4),
            "ECE_After_Calibration": round(ece_clean_cal, 4),
            "ECE_Reduction_Pct": round((ece_clean_raw - ece_clean_cal) / max(ece_clean_raw, 1e-4) * 100, 2),
            "Accuracy": round(accuracy_score(yte_clean, probs_clean_cal.argmax(1)), 4)
        },
        {
            "Dataset": "Noisy Test (Held-out NEFT/IMPS)",
            "Model": "Canonicalized LogReg",
            "Optimal_Temperature_T": round(T, 4),
            "ECE_Before_Calibration": round(ece_noisy_raw, 4),
            "ECE_After_Calibration": round(ece_noisy_cal, 4),
            "ECE_Reduction_Pct": round((ece_noisy_raw - ece_noisy_cal) / max(ece_noisy_raw, 1e-4) * 100, 2),
            "Accuracy": round(accuracy_score(yte_noisy, probs_noisy_cal.argmax(1)), 4)
        }
    ]
    calib_df = pd.DataFrame(calib_records)
    calib_df.to_csv(RESULTS_DIR / "temperature_scaling_calibration.csv", index=False)
    print("Saved results/temperature_scaling_calibration.csv")
    print(calib_df)

    # -------------------------------------------------------------
    # 2e. LABEL-NOISE AUDIT (5-Fold Out-of-Fold Disagreement Analysis)
    # -------------------------------------------------------------
    print("\n--- Running 2e: Label-Noise & Theoretical Accuracy Ceiling Audit ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ytr_clean = le.transform(train_clean.category)
    train_clean_arr = train_clean["transaction_description"].values
    oof_preds = np.zeros(len(train_clean), dtype=int)
    oof_probs = np.zeros(len(train_clean), dtype=float)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_clean_arr, ytr_clean)):
        fold_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
        X_fold_tr = fold_tfidf.fit_transform(train_clean_arr[trn_idx])
        X_fold_va = fold_tfidf.transform(train_clean_arr[val_idx])

        lr_fold = LogisticRegression(C=10, max_iter=300, random_state=42)
        lr_fold.fit(X_fold_tr, ytr_clean[trn_idx])
        proba_val = lr_fold.predict_proba(X_fold_va)

        oof_preds[val_idx] = proba_val.argmax(axis=1)
        oof_probs[val_idx] = proba_val.max(axis=1)

    oof_acc = accuracy_score(ytr_clean, oof_preds)
    # High-confidence disagreements: model confident >= 0.90 but label differs
    high_conf_disagree = (oof_preds != ytr_clean) & (oof_probs >= 0.90)
    num_suspect = int(high_conf_disagree.sum())
    pct_suspect = (num_suspect / len(train_clean)) * 100

    estimated_ceiling = 100.0 - pct_suspect
    print(f"5-Fold OOF Accuracy on Clean Training Set: {oof_acc:.4f}")
    print(f"High-confidence label disagreements (Prob >= 0.90): {num_suspect} / {len(train_clean)} ({pct_suspect:.2f}%)")
    print(f"Estimated empirical accuracy ceiling: ~{estimated_ceiling:.2f}%")

    audit_records = [
        {
            "Audit_Method": "5-Fold Out-of-Fold Cross-Validated Disagreement",
            "Training_Samples_Audited": len(train_clean),
            "OOF_Accuracy": round(oof_acc, 4),
            "High_Confidence_Disagreements_N": num_suspect,
            "Suspected_Label_Noise_Pct": round(pct_suspect, 3),
            "Estimated_Theoretical_Ceiling_Pct": round(estimated_ceiling, 2),
            "Notes": "Samples where model confidence >= 0.90 but disagrees with ground-truth label."
        }
    ]
    audit_df = pd.DataFrame(audit_records)
    audit_df.to_csv(RESULTS_DIR / "label_noise_audit.csv", index=False)
    print("Saved results/label_noise_audit.csv")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
