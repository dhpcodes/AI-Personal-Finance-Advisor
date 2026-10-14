"""Matched-Configuration Comparison and Rigorous McNemar Tests (Task 1 & Task 3).

Specifications:
- SAME training data: mitulshah train split + indian_merchants_train_seed.csv (N=72,947)
- SAME 35% noise augmentation: apply_single_noise(t, rng, rng.choice(TRAIN_KINDS)) with seed 42
- SAME canonicalization + preprocessing: TextPreprocessor(use_canonicalization=True)
- SAME 3,000 features (char_wb 2-4, min_df=5, sublinear_tf=True) for:
    1. LinearSVC (tuned C on val, sigmoid calibrated)
    2. MLPClassifier (256 hidden units, ReLU, early_stopping=True, tuned alpha on val)
    3. LightGBM (100 trees, lr=0.20, num_leaves=63)
- Feature capacity control:
    4. LinearSVC with original 20,000 features (char_wb 2-5, C=10.0, sigmoid calibrated)
- Evaluated on:
    - Clean test (N=20,040)
    - Noisy test V2 (N=20,040)
    - Typo set: both exact N=2,492 and Bernoulli N=2,499 (explains discrepancy)
    - Indian statements (N=105, N=96 non-overlapping, N=58 fully disjoint)
    - PRIMARY: Model-only accuracy (no rent rule)
    - SECONDARY: + Rent rule accuracy
- McNemar tests:
    - Corrected leakage-free SVM (20k feats) vs MiniLM & LightGBM on Indian N=96
    - Pairwise McNemar between matched models on clean test and noisy V2
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from src.preprocessing import TextPreprocessor
from src.predict import TransactionPredictor
from src.model_comparison_suite import with_rent_rule

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"
NOISY_V2_PATH = DATA_DIR / "noisy_test_v2.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"
CLEAN_SEED_PATH = DATA_DIR / "indian_merchants_train_seed_leakage_free.csv"

TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]
OVERLAPPING_CORES = [
    "amazon pay", "dominos pizza", "flipkart internet", "myntra designs",
    "ola cabs", "swiggy", "uber india", "zomato"
]

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

def apply_typo_noise(text: str, rng: random.Random) -> str:
    s = list(text)
    if len(s) > 4:
        i = rng.randint(1, len(s) - 2)
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)

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

def measure_single_latency(model, vectorizer, classes, texts, n_runs=200):
    preprocessor = TextPreprocessor(use_canonicalization=True)
    # Warmup
    for i in range(20):
        c = preprocessor.clean_text(texts[i % len(texts)])
        x = vectorizer.transform([c])
        _ = model.predict(x)
    t0 = time.perf_counter()
    for i in range(n_runs):
        c = preprocessor.clean_text(texts[i % len(texts)])
        x = vectorizer.transform([c])
        _ = model.predict(x)
    tot = time.perf_counter() - t0
    return (tot / n_runs) * 1000.0

def main():
    print("=" * 75)
    print("TASK 1 & 3: MATCHED CONFIGURATION COMPARISON (35% NOISE, SAME FEATS)")
    print("=" * 75)

    # 1. Load Data
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)

    train_df = pd.concat([train_base, df_seed], ignore_index=True)
    print(f"Full Training Set: {len(train_df):,} rows")

    # 2. Apply EXACT 35% Training Noise Augmentation (seed=42)
    print("\n[Step 1] Applying 35% synthetic training noise augmentation (seed=42)...")
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train_df)) < 0.35
    train_noisy = [
        apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) if is_n else t
        for is_n, t in zip(m_tr, train_df["transaction_description"])
    ]

    # Preprocess with canonicalization
    preprocessor = TextPreprocessor(use_canonicalization=True)
    print("  Preprocessing training, validation, and test sets...")
    train_clean = [preprocessor.clean_text(t) for t in train_noisy]
    val_clean = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    test_clean = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]

    le = LabelEncoder().fit(train_df["category"])
    classes = list(le.classes_)
    y_tr = le.transform(train_df["category"])
    y_va = le.transform(val_df["category"])
    y_te = le.transform(test_df["category"])

    # Load Other Test Sets
    df_noisy_v2 = pd.read_csv(NOISY_V2_PATH)
    noisy_v2_clean = [preprocessor.clean_text(t) for t in df_noisy_v2["transaction_description"]]
    y_noisy_v2 = le.transform(df_noisy_v2["category"])

    # Typo Sets: Exact N=2,492 vs Bernoulli N=2,499
    # Exact 2,492:
    rng_exact = random.Random(7)
    exact_typo_indices = sorted(rng_exact.sample(range(len(test_df)), 2492))
    df_typo_exact = test_df.iloc[exact_typo_indices].copy()
    rng_t1 = random.Random(7)
    df_typo_exact["transaction_description"] = [apply_typo_noise(t, rng_t1) for t in df_typo_exact["transaction_description"]]
    typo_exact_clean = [preprocessor.clean_text(t) for t in df_typo_exact["transaction_description"]]
    y_typo_exact = le.transform(df_typo_exact["category"])

    # Bernoulli 2,499 (for exact replication of evaluate_non_linear_comparison):
    rng_bern = random.Random(7)
    bern_typo_indices = [i for i, _ in enumerate(test_df["transaction_description"]) if rng_bern.random() < (2492 / len(test_df))]
    df_typo_bern = test_df.iloc[bern_typo_indices].copy()
    rng_t2 = random.Random(7)
    df_typo_bern["transaction_description"] = [apply_typo_noise(t, rng_t2) for t in df_typo_bern["transaction_description"]]
    typo_bern_clean = [preprocessor.clean_text(t) for t in df_typo_bern["transaction_description"]]
    y_typo_bern = le.transform(df_typo_bern["category"])

    # Indian Sets
    df_own = pd.read_csv(OWN_TEST_PATH)
    df_own["canon"] = df_own["transaction_description"].apply(canonicalize)
    is_ov_9 = df_own["canon"].apply(lambda c: any(core in c.lower() for core in OVERLAPPING_CORES))
    df_own_96 = df_own[~is_ov_9].copy()

    df_clean_seed = pd.read_csv(CLEAN_SEED_PATH)
    clean_seed_cores = set(df_clean_seed["core"].dropna().str.lower().unique())
    is_in_clean_seed = df_own["canon"].apply(lambda c: any(sc in c.lower() or c.lower() in sc for sc in clean_seed_cores))
    df_own_58 = df_own[~is_in_clean_seed].copy()

    own_105_clean = [preprocessor.clean_text(t) for t in df_own["transaction_description"]]
    own_96_clean = [preprocessor.clean_text(t) for t in df_own_96["transaction_description"]]
    own_58_clean = [preprocessor.clean_text(t) for t in df_own_58["transaction_description"]]

    y_own_105 = df_own["category"].tolist()
    y_own_96 = df_own_96["category"].tolist()
    y_own_58 = df_own_58["category"].tolist()

    # 3. Fit 3,000 TF-IDF Features (char_wb 2-4)
    print("\n[Step 2] Fitting 3,000 TF-IDF features (char_wb 2-4, min_df=5)...")
    vec_3k = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=5,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    t0 = time.time()
    X_tr_3k = vec_3k.fit_transform(train_clean)
    X_va_3k = vec_3k.transform(val_clean)
    X_te_3k = vec_3k.transform(test_clean)
    X_nv2_3k = vec_3k.transform(noisy_v2_clean)
    X_typ_ex_3k = vec_3k.transform(typo_exact_clean)
    X_typ_be_3k = vec_3k.transform(typo_bern_clean)
    X_o105_3k = vec_3k.transform(own_105_clean)
    X_o96_3k = vec_3k.transform(own_96_clean)
    X_o58_3k = vec_3k.transform(own_58_clean)
    print(f"  3k TF-IDF fitted in {time.time()-t0:.2f}s | Shape: {X_tr_3k.shape}")

    models_to_test = {}

    # MODEL 1: LinearSVC (Matched 3k)
    print("\n[Step 3] Tuning & Training LinearSVC on Matched 3k Features...")
    best_c_lin = None
    best_val_lin = -1.0
    best_lin_svc = None
    for c_val in [0.1, 1.0, 5.0, 10.0]:
        svc = LinearSVC(C=c_val, random_state=42, max_iter=2000)
        svc.fit(X_tr_3k, y_tr)
        v_acc = accuracy_score(y_va, svc.predict(X_va_3k))
        print(f"  LinearSVC(C={c_val}): Val Acc={v_acc*100:.2f}%")
        if v_acc > best_val_lin:
            best_val_lin = v_acc
            best_c_lin = c_val
            best_lin_svc = svc

    print(f"  Calibrating best LinearSVC(C={best_c_lin})...")
    t0 = time.time()
    cal_lin_3k = CalibratedClassifierCV(best_lin_svc, method="sigmoid")
    cal_lin_3k.fit(X_tr_3k, y_tr)
    dur_lin_3k = time.time() - t0
    models_to_test["Matched_Linear_SVM_3k"] = {
        "model": cal_lin_3k,
        "vectorizer": vec_3k,
        "features": 3000,
        "fit_time": dur_lin_3k,
        "save_path": MODELS_DIR / "matched_linear_svm_3k.joblib"
    }

    # MODEL 2: MLP (Matched 3k)
    print("\n[Step 4] Tuning & Training MLP on Matched 3k Features...")
    best_alpha = None
    best_val_mlp = -1.0
    best_mlp = None
    best_mlp_dur = 0.0
    for alpha_val in [1e-4, 1e-3, 1e-2]:
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=alpha_val,
            batch_size=256,
            learning_rate_init=0.001,
            early_stopping=True,
            n_iter_no_change=5,
            max_iter=50,
            random_state=42,
            verbose=False
        )
        t_m0 = time.time()
        mlp.fit(X_tr_3k, y_tr)
        dur = time.time() - t_m0
        v_acc = accuracy_score(y_va, mlp.predict(X_va_3k))
        print(f"  MLP(alpha={alpha_val}): Fit={dur:.1f}s | Epochs={mlp.n_iter_} | Val Acc={v_acc*100:.2f}%")
        if v_acc > best_val_mlp:
            best_val_mlp = v_acc
            best_alpha = alpha_val
            best_mlp = mlp
            best_mlp_dur = dur

    models_to_test["Matched_MLP_3k"] = {
        "model": best_mlp,
        "vectorizer": vec_3k,
        "features": 3000,
        "fit_time": best_mlp_dur,
        "save_path": MODELS_DIR / "matched_mlp_3k.joblib"
    }

    # MODEL 3: LightGBM (Matched 3k)
    print("\n[Step 5] Training LightGBM on Matched 3k Features...")
    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.20,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    t0 = time.time()
    lgbm.fit(X_tr_3k, y_tr)
    dur_lgb = time.time() - t0
    v_acc_lgb = accuracy_score(y_va, lgbm.predict(X_va_3k))
    print(f"  LightGBM: Fit={dur_lgb:.1f}s | Val Acc={v_acc_lgb*100:.2f}%")
    models_to_test["Matched_LightGBM_3k"] = {
        "model": lgbm,
        "vectorizer": vec_3k,
        "features": 3000,
        "fit_time": dur_lgb,
        "save_path": MODELS_DIR / "matched_lightgbm_3k.joblib"
    }

    # MODEL 4: LinearSVC (Original 20k, char_wb 2-5)
    print("\n[Step 6] Fitting 20,000 TF-IDF features and Training Original SVM Architecture...")
    vec_20k = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=True
    )
    t0 = time.time()
    X_tr_20k = vec_20k.fit_transform(train_clean)
    X_va_20k = vec_20k.transform(val_clean)
    X_te_20k = vec_20k.transform(test_clean)
    X_nv2_20k = vec_20k.transform(noisy_v2_clean)
    X_typ_ex_20k = vec_20k.transform(typo_exact_clean)
    X_typ_be_20k = vec_20k.transform(typo_bern_clean)
    X_o105_20k = vec_20k.transform(own_105_clean)
    X_o96_20k = vec_20k.transform(own_96_clean)
    X_o58_20k = vec_20k.transform(own_58_clean)
    print(f"  20k TF-IDF fitted in {time.time()-t0:.2f}s | Shape: {X_tr_20k.shape}")

    t0 = time.time()
    base_svc_20k = LinearSVC(C=10.0, max_iter=2000, random_state=42)
    cal_lin_20k = CalibratedClassifierCV(base_svc_20k, method="sigmoid")
    cal_lin_20k.fit(X_tr_20k, y_tr)
    dur_lin_20k = time.time() - t0
    v_acc_20k = accuracy_score(y_va, cal_lin_20k.predict(X_va_20k))
    print(f"  LinearSVC (20k feats, C=10.0): Fit={dur_lin_20k:.1f}s | Val Acc={v_acc_20k*100:.2f}%")
    models_to_test["Linear_SVM_Original_20k"] = {
        "model": cal_lin_20k,
        "vectorizer": vec_20k,
        "features": 20000,
        "fit_time": dur_lin_20k,
        "save_path": MODELS_DIR / "matched_linear_svm_20k.joblib"
    }

    # Save models
    for m_name, m_info in models_to_test.items():
        joblib.dump({
            "version": "1.0.0-matched_experiment",
            "model_name": m_name,
            "vectorizer": m_info["vectorizer"],
            "classes": classes,
            "model": m_info["model"],
            "features": m_info["features"],
            "fit_time_sec": round(m_info["fit_time"], 2)
        }, m_info["save_path"])
        print(f"  [SAVED] {m_info['save_path'].name}")

    # 4. Standardized Evaluation on All Splits
    print("\n[Step 7] Evaluating All Models Across Clean, Noisy V2, Typo, and Indian Sets...")
    eval_results = []
    clean_preds_dict = {}
    nv2_preds_dict = {}
    indian_96_raw_dict = {}
    indian_96_rent_dict = {}

    for m_name, m_info in models_to_test.items():
        mdl = m_info["model"]
        vec = m_info["vectorizer"]
        is_20k = (m_info["features"] == 20000)

        # Transforms
        xt_tr = X_tr_20k if is_20k else X_tr_3k
        xt_va = X_va_20k if is_20k else X_va_3k
        xt_te = X_te_20k if is_20k else X_te_3k
        xt_nv2 = X_nv2_20k if is_20k else X_nv2_3k
        xt_ex = X_typ_ex_20k if is_20k else X_typ_ex_3k
        xt_be = X_typ_be_20k if is_20k else X_typ_be_3k
        xt_105 = X_o105_20k if is_20k else X_o105_3k
        xt_96 = X_o96_20k if is_20k else X_o96_3k
        xt_58 = X_o58_20k if is_20k else X_o58_3k

        # Predictions
        p_tr = mdl.predict(xt_tr)
        p_va = mdl.predict(xt_va)
        p_te = mdl.predict(xt_te)
        p_nv2 = mdl.predict(xt_nv2)
        p_ex = mdl.predict(xt_ex)
        p_be = mdl.predict(xt_be)

        acc_tr = accuracy_score(y_tr, p_tr)
        acc_va = accuracy_score(y_va, p_va)
        acc_te = accuracy_score(y_te, p_te)
        f1_te = f1_score(y_te, p_te, average="macro")
        acc_nv2 = accuracy_score(y_noisy_v2, p_nv2)
        f1_nv2 = f1_score(y_noisy_v2, p_nv2, average="macro")
        acc_ex = accuracy_score(y_typo_exact, p_ex)
        acc_be = accuracy_score(y_typo_bern, p_be)

        ci_c_low, ci_c_high = wilson_score_interval(acc_te, len(y_te))
        ci_n_low, ci_n_high = wilson_score_interval(acc_nv2, len(y_noisy_v2))

        clean_preds_dict[m_name] = p_te
        nv2_preds_dict[m_name] = p_nv2

        # Indian sets: Model-Only (Raw) vs +Rent Rule
        # 105
        p105_raw = mdl.predict(xt_105)
        cat_105_raw = [classes[i] for i in p105_raw]
        cat_105_rent = with_rent_rule(df_own["transaction_description"].tolist(), cat_105_raw)
        acc_105_raw = accuracy_score(y_own_105, cat_105_raw)
        acc_105_rent = accuracy_score(y_own_105, cat_105_rent)

        # 96
        p96_raw = mdl.predict(xt_96)
        cat_96_raw = [classes[i] for i in p96_raw]
        cat_96_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), cat_96_raw)
        acc_96_raw = accuracy_score(y_own_96, cat_96_raw)
        acc_96_rent = accuracy_score(y_own_96, cat_96_rent)
        indian_96_raw_dict[m_name] = cat_96_raw
        indian_96_rent_dict[m_name] = cat_96_rent

        # 58
        p58_raw = mdl.predict(xt_58)
        cat_58_raw = [classes[i] for i in p58_raw]
        cat_58_rent = with_rent_rule(df_own_58["transaction_description"].tolist(), cat_58_raw)
        acc_58_raw = accuracy_score(y_own_58, cat_58_raw)
        acc_58_rent = accuracy_score(y_own_58, cat_58_rent)

        # Latency
        lat = measure_single_latency(mdl, vec, classes, test_df["transaction_description"].tolist(), n_runs=200)

        eval_results.append({
            "Model": m_name,
            "Features": m_info["features"],
            "Train_Acc": round(acc_tr, 4),
            "Val_Acc": round(acc_va, 4),
            "Clean_Test_Acc": round(acc_te, 4),
            "Clean_95_CI": f"[{ci_c_low*100:.2f}%, {ci_c_high*100:.2f}%]",
            "Clean_Macro_F1": round(f1_te, 4),
            "Noisy_V2_Acc": round(acc_nv2, 4),
            "Noisy_V2_95_CI": f"[{ci_n_low*100:.2f}%, {ci_n_high*100:.2f}%]",
            "Noisy_V2_Macro_F1": round(f1_nv2, 4),
            "Typo_Exact_2492_Acc": round(acc_ex, 4),
            "Typo_Bern_2499_Acc": round(acc_be, 4),
            "Indian_96_ModelOnly_Acc": round(acc_96_raw, 4),
            "Indian_96_PlusRent_Acc": round(acc_96_rent, 4),
            "Indian_58_ModelOnly_Acc": round(acc_58_raw, 4),
            "Indian_58_PlusRent_Acc": round(acc_58_rent, 4),
            "Indian_105_ModelOnly_Acc": round(acc_105_raw, 4),
            "Indian_105_PlusRent_Acc": round(acc_105_rent, 4),
            "Single_Latency_ms": round(lat, 3),
            "Fit_Time_Sec": round(m_info["fit_time"], 2)
        })

    df_out = pd.DataFrame(eval_results)
    out_csv = RESULTS_DIR / "matched_configuration_comparison.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Matched comparison saved to {out_csv}")
    print(df_out.to_string(index=False))

    # 5. TASK 3: McNemar Significance Tests
    print("\n[Step 8] Running McNemar Significance Tests (Task 3)...")
    mcnemar_out = {}

    # Pairwise between matched models on Clean Test and Noisy V2
    pairs = [
        ("Matched_Linear_SVM_3k", "Matched_MLP_3k"),
        ("Matched_Linear_SVM_3k", "Matched_LightGBM_3k"),
        ("Matched_MLP_3k", "Matched_LightGBM_3k"),
        ("Linear_SVM_Original_20k", "Matched_Linear_SVM_3k"),
        ("Linear_SVM_Original_20k", "Matched_LightGBM_3k"),
        ("Linear_SVM_Original_20k", "Matched_MLP_3k")
    ]
    for ma, mb in pairs:
        # Clean
        res_clean = mcnemar_test(y_te, clean_preds_dict[ma], clean_preds_dict[mb])
        mcnemar_out[f"{ma}_vs_{mb}_CleanTest"] = res_clean
        # Noisy V2
        res_nv2 = mcnemar_test(y_noisy_v2, nv2_preds_dict[ma], nv2_preds_dict[mb])
        mcnemar_out[f"{ma}_vs_{mb}_NoisyV2"] = res_nv2
        print(f"  {ma} vs {mb}: Clean chi2={res_clean['statistic']}, p={res_clean['p_value']} | Noisy chi2={res_nv2['statistic']}, p={res_nv2['p_value']}")

    # Corrected Leakage-Free SVM (20k feats) vs MiniLM & LightGBM on Indian N=96
    print("\n[Step 9] Evaluating Corrected Leakage-Free SVM vs MiniLM & LightGBM on Indian N=96...")
    art_leakfree = joblib.load(MODELS_DIR / "variant_b_candidate_leakage_free_20k.joblib")
    pred_lf = TransactionPredictor(model_path=MODELS_DIR / "variant_b_candidate_leakage_free_20k.joblib")
    pred_minilm = TransactionPredictor(model_path=MODELS_DIR / "minilm_lr_transaction_classifier.joblib")
    pred_lgb = TransactionPredictor(model_path=MODELS_DIR / "lightgbm_transaction_classifier.joblib")

    # Get predictions on df_own_96
    res_lf_96 = pred_lf.predict_batch(df_own_96[["transaction_description"]])
    p_lf_raw = res_lf_96["predicted_category"].tolist()
    p_lf_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), p_lf_raw)

    res_minilm_96 = pred_minilm.predict_batch(df_own_96[["transaction_description"]])
    p_minilm_raw = res_minilm_96["predicted_category"].tolist()
    p_minilm_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), p_minilm_raw)

    res_lgb_96 = pred_lgb.predict_batch(df_own_96[["transaction_description"]])
    p_lgb_raw = res_lgb_96["predicted_category"].tolist()
    p_lgb_rent = with_rent_rule(df_own_96["transaction_description"].tolist(), p_lgb_raw)

    # McNemar on N=96: Model-Only
    mc_lf_minilm_raw = mcnemar_test(y_own_96, p_lf_raw, p_minilm_raw)
    mc_lf_lgb_raw = mcnemar_test(y_own_96, p_lf_raw, p_lgb_raw)
    mcnemar_out["LeakageFree_20k_vs_MiniLM_Indian96_ModelOnly"] = mc_lf_minilm_raw
    mcnemar_out["LeakageFree_20k_vs_LightGBM_Indian96_ModelOnly"] = mc_lf_lgb_raw

    # McNemar on N=96: +Rent Rule
    mc_lf_minilm_rent = mcnemar_test(y_own_96, p_lf_rent, p_minilm_rent)
    mc_lf_lgb_rent = mcnemar_test(y_own_96, p_lf_rent, p_lgb_rent)
    mcnemar_out["LeakageFree_20k_vs_MiniLM_Indian96_PlusRent"] = mc_lf_minilm_rent
    mcnemar_out["LeakageFree_20k_vs_LightGBM_Indian96_PlusRent"] = mc_lf_lgb_rent

    print(f"  LeakageFree 20k vs MiniLM (Indian 96 Model-Only): chi2={mc_lf_minilm_raw['statistic']}, p={mc_lf_minilm_raw['p_value']}")
    print(f"  LeakageFree 20k vs LightGBM (Indian 96 Model-Only): chi2={mc_lf_lgb_raw['statistic']}, p={mc_lf_lgb_raw['p_value']}")
    print(f"  LeakageFree 20k vs MiniLM (Indian 96 +Rent): chi2={mc_lf_minilm_rent['statistic']}, p={mc_lf_minilm_rent['p_value']}")
    print(f"  LeakageFree 20k vs LightGBM (Indian 96 +Rent): chi2={mc_lf_lgb_rent['statistic']}, p={mc_lf_lgb_rent['p_value']}")

    with open(RESULTS_DIR / "mcnemar_tests_matched_and_corrected.json", "w") as f:
        json.dump(mcnemar_out, f, indent=2)
    print(f"[SAVED] McNemar results saved to {RESULTS_DIR / 'mcnemar_tests_matched_and_corrected.json'}")

if __name__ == "__main__":
    main()
