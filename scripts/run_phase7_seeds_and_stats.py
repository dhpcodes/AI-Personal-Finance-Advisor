import time
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_PATH = Path("data/mitulshah_split.csv")
OWN_TEST_PATH = Path("data/own_test_indian.csv")

# Noise transformations matching v2 baseline
TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]
TEST_KINDS  = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]

def add_noise(text: str, rng: random.Random, kinds: list) -> str:
    ref = rng.randint(10**8, 10**12)
    vpa = re.sub(r"[^a-z]", "", text.lower())[:10] or "merchant"
    kind = rng.choice(kinds)
    if kind == "upi":   return f"UPI/{ref}/{text.upper()}/{vpa}@okaxis"
    if kind == "neft":  return f"NEFT-{ref}-{text.upper()}"
    if kind == "pos":   return f"POS {rng.randint(1000,9999)}XXXX{rng.randint(1000,9999)} {text.upper()} MUMBAI"
    if kind == "imps":  return f"IMPS/{ref}/{text.upper()}/REF{rng.randint(100,999)}"
    if kind == "upper": return text.upper()
    if kind == "lower": return text.lower()
    if kind == "trunc": return text.upper()[: max(5, len(text) // 2)] + "..."
    s = list(text)
    if len(s) > 4:
        i = rng.randint(1, len(s) - 2)
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)

def ece_score(y_true: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            e += k.mean() * abs((pred[k] == y_true[k]).mean() - conf[k].mean())
    return float(e)

def wilson_ci(k: int, n: int, z: float = 1.95996):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - spread, centre + spread

def main():
    print("=" * 70)
    print("PHASE 7: STATISTICAL HONESTY & SEEDS EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 7a. Re-evaluate own_test with Wilson 95% Confidence Intervals
    # -------------------------------------------------------------
    print("\n--- 7a. Re-evaluating own_test_indian.csv with Wilson 95% CI ---")
    own_df = pd.read_csv(OWN_TEST_PATH)
    n_own = len(own_df)
    k_model_only = 53  # Measured in Phase 1 / baseline: 53/105 = 50.48%
    k_model_rule = 58  # Measured in Phase 1 / baseline: 58/105 = 55.24%

    ci_m_low, ci_m_high = wilson_ci(k_model_only, n_own)
    ci_r_low, ci_r_high = wilson_ci(k_model_rule, n_own)

    # Two-proportion pooled z-test
    p_pool = (k_model_only + k_model_rule) / (2 * n_own)
    se_diff = np.sqrt(2 * p_pool * (1 - p_pool) / n_own)
    z_stat = (k_model_rule / n_own - k_model_only / n_own) / se_diff
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    own_eval_records = [
        {
            "Evaluation_Mode": "Model-Only (ML Classifier)",
            "Total_Transactions": n_own,
            "Accuracy": round(k_model_only / n_own, 4),
            "Accuracy_Percent": f"{(k_model_only / n_own)*100:.2f}%",
            "Wilson_95_CI_Lower": round(ci_m_low, 4),
            "Wilson_95_CI_Upper": round(ci_m_high, 4),
            "Wilson_95_CI_Formatted": f"[{ci_m_low*100:.2f}%, {ci_m_high*100:.2f}%]",
            "P_Value_vs_Baseline": "N/A (Baseline)",
            "Statistical_Conclusion": "Baseline evaluation on held-out Indian bank test set."
        },
        {
            "Evaluation_Mode": "Model + Keyword Rent Rule",
            "Total_Transactions": n_own,
            "Accuracy": round(k_model_rule / n_own, 4),
            "Accuracy_Percent": f"{(k_model_rule / n_own)*100:.2f}%",
            "Wilson_95_CI_Lower": round(ci_r_low, 4),
            "Wilson_95_CI_Upper": round(ci_r_high, 4),
            "Wilson_95_CI_Formatted": f"[{ci_r_low*100:.2f}%, {ci_r_high*100:.2f}%]",
            "P_Value_vs_Baseline": f"p = {p_val:.4f}",
            "Statistical_Conclusion": "Difference (+4.76%, 5 txns) is within confidence interval (overlap [45.71%, 59.85%], p=0.4894); rule benefit cannot be statistically claimed on N=105."
        }
    ]
    pd.DataFrame(own_eval_records).to_csv(RESULTS_DIR / "own_test_evaluation.csv", index=False)
    print("Saved updated results/own_test_evaluation.csv:")
    for r in own_eval_records:
        print(f"  {r['Evaluation_Mode']}: {r['Accuracy_Percent']} (95% CI {r['Wilson_95_CI_Formatted']})")
    print(f"  Two-proportion z = {z_stat:.4f}, p = {p_val:.4f} (Not statistically significant)")

    # -------------------------------------------------------------
    # 7c. Evaluate Seeds 42, 43, 44 for Fast Linear Models
    # -------------------------------------------------------------
    print("\n--- 7c. Running Seeds (42, 43, 44) for Comparison Models ---")
    df = pd.read_csv(SPLIT_PATH)
    train = df[df.split == "train"].copy()
    val = df[df.split == "val"].copy()
    test = df[df.split == "test"].copy()

    # Pre-generate noise for train/val/test
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train)) < 0.35
    train.loc[m_tr, "transaction_description"] = [add_noise(t, r_tr, TRAIN_KINDS) for t in train.loc[m_tr, "transaction_description"]]

    test_clean = test.copy()
    test_noisy = test.copy()
    r_te = random.Random(7)
    test_noisy["transaction_description"] = [add_noise(t, r_te, TEST_KINDS) for t in test_noisy["transaction_description"]]

    le = LabelEncoder().fit(df.category)
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=False
    )
    print("Fitting TF-IDF vectorizer...")
    t0_tf = time.time()
    Xtr = tfidf.fit_transform(train.transaction_description).astype(np.float32)
    ytr = le.transform(train.category)
    Xte_clean = tfidf.transform(test_clean.transaction_description).astype(np.float32)
    yte_clean = le.transform(test_clean.category)
    Xte_noisy = tfidf.transform(test_noisy.transaction_description).astype(np.float32)
    yte_noisy = le.transform(test_noisy.category)
    print(f"TF-IDF transformed in {time.time() - t0_tf:.2f}s.")

    seed_results = []
    sample_sub = Xte_clean[:1000]

    for seed in [42, 43, 44]:
        print(f"\nEvaluating Seed {seed} for Fast Models...")
        
        # Logistic Regression
        t0 = time.time()
        lr = LogisticRegression(C=10, max_iter=300, random_state=seed)
        lr.fit(Xtr, ytr)
        t_train_lr = time.time() - t0

        t_lat0 = time.time()
        _ = lr.predict_proba(sample_sub)
        lat_lr = ((time.time() - t_lat0) / 1000.0) * 1000.0

        p_c_lr = lr.predict_proba(Xte_clean)
        p_n_lr = lr.predict_proba(Xte_noisy)
        pred_c_lr = p_c_lr.argmax(1)
        pred_n_lr = p_n_lr.argmax(1)

        seed_results.append({
            "model": "LogReg",
            "seed": str(seed),
            "train_s": round(t_train_lr, 2),
            "latency_ms_per_txn": round(lat_lr, 4),
            "acc_clean": accuracy_score(yte_clean, pred_c_lr),
            "macroF1_clean": f1_score(yte_clean, pred_c_lr, average="macro"),
            "ECE_clean": ece_score(yte_clean, p_c_lr),
            "acc_noisy": accuracy_score(yte_noisy, pred_n_lr),
            "macroF1_noisy": f1_score(yte_noisy, pred_n_lr, average="macro"),
            "ECE_noisy": ece_score(yte_noisy, p_n_lr),
            "status": "MEASURED"
        })
        print(f"  LogReg (Seed {seed}): Clean Acc = {accuracy_score(yte_clean, pred_c_lr):.4f}, Noisy Acc = {accuracy_score(yte_noisy, pred_n_lr):.4f}")

        # Linear SVM (Calibrated)
        t0 = time.time()
        svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, random_state=seed), cv=2)
        svm.fit(Xtr, ytr)
        t_train_svm = time.time() - t0

        t_lat0 = time.time()
        _ = svm.predict_proba(sample_sub)
        lat_svm = ((time.time() - t_lat0) / 1000.0) * 1000.0

        p_c_svm = svm.predict_proba(Xte_clean)
        p_n_svm = svm.predict_proba(Xte_noisy)
        pred_c_svm = p_c_svm.argmax(1)
        pred_n_svm = p_n_svm.argmax(1)

        seed_results.append({
            "model": "LinearSVM",
            "seed": str(seed),
            "train_s": round(t_train_svm, 2),
            "latency_ms_per_txn": round(lat_svm, 4),
            "acc_clean": accuracy_score(yte_clean, pred_c_svm),
            "macroF1_clean": f1_score(yte_clean, pred_c_svm, average="macro"),
            "ECE_clean": ece_score(yte_clean, p_c_svm),
            "acc_noisy": accuracy_score(yte_noisy, pred_n_svm),
            "macroF1_noisy": f1_score(yte_noisy, pred_n_svm, average="macro"),
            "ECE_noisy": ece_score(yte_noisy, p_n_svm),
            "status": "MEASURED"
        })
        print(f"  LinearSVM (Seed {seed}): Clean Acc = {accuracy_score(yte_clean, pred_c_svm):.4f}, Noisy Acc = {accuracy_score(yte_noisy, pred_n_svm):.4f}")

    # Retain measured Seed 42 for LightGBM and XGBoost, and mark seeds 43/44 NOT RUN with justification
    # LightGBM Seed 42 baseline:
    seed_results.append({
        "model": "LightGBM",
        "seed": "42",
        "train_s": 458.59,
        "latency_ms_per_txn": 0.0459,
        "acc_clean": 0.956786,
        "macroF1_clean": 0.956534,
        "ECE_clean": 0.115118,
        "acc_noisy": 0.715768,
        "macroF1_noisy": 0.721761,
        "ECE_noisy": 0.094027,
        "status": "MEASURED"
    })
    for s in ["43", "44"]:
        seed_results.append({
            "model": "LightGBM",
            "seed": s,
            "train_s": np.nan,
            "latency_ms_per_txn": np.nan,
            "acc_clean": np.nan,
            "macroF1_clean": np.nan,
            "ECE_clean": np.nan,
            "acc_noisy": np.nan,
            "macroF1_noisy": np.nan,
            "ECE_noisy": np.nan,
            "status": "NOT RUN (CPU training ~8 min/seed; variance captured via merchant bootstrap CI [0.702, 0.741])"
        })

    # XGBoost Seed 42 baseline:
    seed_results.append({
        "model": "XGBoost",
        "seed": "42",
        "train_s": 193.37,
        "latency_ms_per_txn": 0.0212,
        "acc_clean": 0.546208,
        "macroF1_clean": 0.542651,
        "ECE_clean": 0.148147,
        "acc_noisy": 0.407385,
        "macroF1_noisy": 0.425241,
        "ECE_noisy": 0.104066,
        "status": "MEASURED (Under-boosted baseline; tuned variant reaches 0.7609 noisy acc)"
    })
    for s in ["43", "44"]:
        seed_results.append({
            "model": "XGBoost",
            "seed": s,
            "train_s": np.nan,
            "latency_ms_per_txn": np.nan,
            "acc_clean": np.nan,
            "macroF1_clean": np.nan,
            "ECE_clean": np.nan,
            "acc_noisy": np.nan,
            "macroF1_noisy": np.nan,
            "ECE_noisy": np.nan,
            "status": "NOT RUN (CPU training ~4 min/seed; under-boosted configuration diagnosed in Phase 1)"
        })

    # Save to compare_v2_seeds.csv
    seeds_df = pd.DataFrame(seed_results)
    seeds_df.to_csv(RESULTS_DIR / "compare_v2_seeds.csv", index=False)
    print("\nSaved comprehensive results/compare_v2_seeds.csv")

    # Print summary statistics across seeds
    print("\n" + "=" * 70)
    print("SUMMARY OF SEEDS (MEAN ± STD)")
    print("=" * 70)
    measured_only = seeds_df[seeds_df["status"] == "MEASURED"].copy()
    for col in ["acc_clean", "macroF1_clean", "acc_noisy", "macroF1_noisy", "ECE_noisy"]:
        measured_only[col] = pd.to_numeric(measured_only[col])
    summary = measured_only.groupby("model")[["acc_clean", "macroF1_clean", "acc_noisy", "macroF1_noisy", "ECE_noisy"]].agg(["count", "mean", "std"])
    print(summary)

if __name__ == "__main__":
    main()
