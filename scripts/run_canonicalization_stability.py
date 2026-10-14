"""Multi-seed stability evaluation of transaction canonicalization variants.

Evaluates Variant A (Raw Baseline), Variant B (Canonicalized), and Variant C (Canonicalized + Engineered)
across seeds 42, 43, 44 on Clean Test, Noisy Test, and Typo Noise subset.
Reports mean ± std for each metric.
"""

import sys
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from scripts.run_canonicalization_experiment import (
    extract_handcrafted_features,
    apply_single_noise,
    TRAIN_KINDS,
    TEST_KINDS
)

SPLIT_PATH = Path("data/mitulshah_split.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STEP 5: MULTI-SEED STABILITY BENCHMARK (SEEDS 42, 43, 44)")
    print("=" * 80)

    df = pd.read_csv(SPLIT_PATH)
    train_df = df[df.split == "train"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)

    le = LabelEncoder().fit(df.category)
    y_tr = le.transform(train_df.category)
    y_te_clean = le.transform(test_df.category)

    # Clean text versions
    test_clean_raw = test_df.transaction_description.tolist()
    test_clean_canon = [canonicalize(t) for t in test_clean_raw]

    SEEDS = [42, 43, 44]
    stability_records = []

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        # 1. Generate training noise for this seed
        r_tr = random.Random(seed)
        m_tr = np.random.RandomState(seed).rand(len(train_df)) < 0.35
        train_noisy = [
            apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) if is_n else t
            for is_n, t in zip(m_tr, train_df.transaction_description)
        ]
        train_canon = [canonicalize(t) for t in train_noisy]

        # 2. Generate noisy test set for this seed
        r_te = random.Random(seed + 100)
        test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
        test_noisy_raw = [
            apply_single_noise(t, r_te, nt) for t, nt in zip(test_df.transaction_description, test_noise_types)
        ]
        test_noisy_canon = [canonicalize(t) for t in test_noisy_raw]
        y_te_noisy = le.transform(test_df.category)

        # Typo subset mask
        typo_mask = np.array([nt == "typo" for nt in test_noise_types])
        y_te_typo = y_te_noisy[typo_mask]

        # -------------------------------------------------------------
        # VARIANT A: Raw Text Baseline
        # -------------------------------------------------------------
        tfidf_a = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
        Xtr_a = tfidf_a.fit_transform(train_noisy)
        Xte_clean_a = tfidf_a.transform(test_clean_raw)
        Xte_noisy_a = tfidf_a.transform(test_noisy_raw)
        Xte_typo_a  = Xte_noisy_a[typo_mask]

        clf_a = LogisticRegression(C=10, max_iter=300, random_state=seed).fit(Xtr_a, y_tr)
        p_c_a = clf_a.predict(Xte_clean_a)
        p_n_a = clf_a.predict(Xte_noisy_a)
        p_t_a = clf_a.predict(Xte_typo_a)

        stability_records.append({
            "variant": "Variant A (Raw Baseline)",
            "seed": seed,
            "clean_acc": accuracy_score(y_te_clean, p_c_a),
            "clean_macroF1": f1_score(y_te_clean, p_c_a, average="macro"),
            "noisy_acc": accuracy_score(y_te_noisy, p_n_a),
            "noisy_macroF1": f1_score(y_te_noisy, p_n_a, average="macro"),
            "typo_acc": accuracy_score(y_te_typo, p_t_a),
            "typo_macroF1": f1_score(y_te_typo, p_t_a, average="macro", zero_division=0)
        })

        # -------------------------------------------------------------
        # VARIANT B: Canonicalized Text
        # -------------------------------------------------------------
        tfidf_b = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
        Xtr_b = tfidf_b.fit_transform(train_canon)
        Xte_clean_b = tfidf_b.transform(test_clean_canon)
        Xte_noisy_b = tfidf_b.transform(test_noisy_canon)
        Xte_typo_b  = Xte_noisy_b[typo_mask]

        clf_b = LogisticRegression(C=10, max_iter=300, random_state=seed).fit(Xtr_b, y_tr)
        p_c_b = clf_b.predict(Xte_clean_b)
        p_n_b = clf_b.predict(Xte_noisy_b)
        p_t_b = clf_b.predict(Xte_typo_b)

        stability_records.append({
            "variant": "Variant B (Canonicalized)",
            "seed": seed,
            "clean_acc": accuracy_score(y_te_clean, p_c_b),
            "clean_macroF1": f1_score(y_te_clean, p_c_b, average="macro"),
            "noisy_acc": accuracy_score(y_te_noisy, p_n_b),
            "noisy_macroF1": f1_score(y_te_noisy, p_n_b, average="macro"),
            "typo_acc": accuracy_score(y_te_typo, p_t_b),
            "typo_macroF1": f1_score(y_te_typo, p_t_b, average="macro", zero_division=0)
        })

        # -------------------------------------------------------------
        # VARIANT C: Canonicalized + Engineered Features
        # -------------------------------------------------------------
        tfidf_c_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True)
        Xtr_word = tfidf_c_word.fit_transform(train_canon)
        Xte_clean_word = tfidf_c_word.transform(test_clean_canon)
        Xte_noisy_word = tfidf_c_word.transform(test_noisy_canon)

        scaler = MaxAbsScaler()
        F_tr = csr_matrix(scaler.fit_transform(extract_handcrafted_features(train_canon)))
        F_clean = csr_matrix(scaler.transform(extract_handcrafted_features(test_clean_canon)))
        F_noisy = csr_matrix(scaler.transform(extract_handcrafted_features(test_noisy_canon)))

        Xtr_c = hstack([Xtr_b, Xtr_word, F_tr]).tocsr()
        Xte_clean_c = hstack([Xte_clean_b, Xte_clean_word, F_clean]).tocsr()
        Xte_noisy_c = hstack([Xte_noisy_b, Xte_noisy_word, F_noisy]).tocsr()
        Xte_typo_c  = Xte_noisy_c[typo_mask]

        clf_c = LogisticRegression(C=10, max_iter=300, random_state=seed).fit(Xtr_c, y_tr)
        p_c_c = clf_c.predict(Xte_clean_c)
        p_n_c = clf_c.predict(Xte_noisy_c)
        p_t_c = clf_c.predict(Xte_typo_c)

        stability_records.append({
            "variant": "Variant C (Canon + Engineered)",
            "seed": seed,
            "clean_acc": accuracy_score(y_te_clean, p_c_c),
            "clean_macroF1": f1_score(y_te_clean, p_c_c, average="macro"),
            "noisy_acc": accuracy_score(y_te_noisy, p_n_c),
            "noisy_macroF1": f1_score(y_te_noisy, p_n_c, average="macro"),
            "typo_acc": accuracy_score(y_te_typo, p_t_c),
            "typo_macroF1": f1_score(y_te_typo, p_t_c, average="macro", zero_division=0)
        })

    stab_df = pd.DataFrame(stability_records)
    stab_df.to_csv(RESULTS_DIR / "canonicalization_seeds_stability.csv", index=False)
    print("\nSaved raw multi-seed records to results/canonicalization_seeds_stability.csv")

    print("\n" + "=" * 80)
    print("STABILITY SUMMARY: MEAN ± STD ACROSS SEEDS (42, 43, 44)")
    print("=" * 80)
    metric_cols = ["clean_acc", "clean_macroF1", "noisy_acc", "noisy_macroF1", "typo_acc", "typo_macroF1"]
    summary = stab_df.groupby("variant")[metric_cols].agg(["mean", "std"]).round(4)
    print(summary)


if __name__ == "__main__":
    main()
