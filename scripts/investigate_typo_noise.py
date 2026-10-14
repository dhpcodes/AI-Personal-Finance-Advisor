"""Investigation of typo noise degradation on Training and Validation splits.

Compares:
  1. Char TF-IDF only (Variant B)
  2. Word TF-IDF only
  3. Char + Word TF-IDF
  4. Char + Handcrafted (7)
  5. Char + Word + Handcrafted (Variant C)
  6. Proposed Improvements:
     - Improvement 1: Char TF-IDF (2-5, 30k) + Sublinear Scaling (Pure character approach)
     - Improvement 2: Char TF-IDF (2-5, 20k) + Word TF-IDF with Typo Noise Augmentation in Training
     - Improvement 3: Char TF-IDF (2-5, 20k) + L2 Regularization tuning (C=1.0 or C=5.0) to prevent word over-reliance
"""

import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonicalize import canonicalize
from scripts.run_canonicalization_experiment import extract_handcrafted_features, apply_single_noise, TRAIN_KINDS

SPLIT_PATH = Path("data/mitulshah_split.csv")


def main():
    print("=" * 80)
    print("STEP 3 & 4: INVESTIGATION OF TYPO NOISE (TRAIN & VALIDATION ONLY)")
    print("=" * 80)

    df = pd.read_csv(SPLIT_PATH)
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df   = df[df.split == "val"].reset_index(drop=True)

    le = LabelEncoder().fit(df.category)
    classes = list(le.classes_)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)

    # 1. Base training set (35% standard noise)
    r_tr = random.Random(42)
    m_tr = np.random.RandomState(42).rand(len(train_df)) < 0.35
    train_noisy = [
        apply_single_noise(t, r_tr, r_tr.choice(TRAIN_KINDS)) if is_n else t
        for is_n, t in zip(m_tr, train_df.transaction_description)
    ]
    train_c = [canonicalize(t) for t in train_noisy]

    # 2. Validation set variants:
    # 2a. Clean validation
    val_clean_c = [canonicalize(t) for t in val_df.transaction_description]
    # 2b. Standard mixed noise validation (35% noise)
    r_va = random.Random(11)
    m_va = np.random.RandomState(1).rand(len(val_df)) < 0.35
    val_mixed_noisy = [
        apply_single_noise(t, r_va, r_va.choice(TRAIN_KINDS)) if is_n else t
        for is_n, t in zip(m_va, val_df.transaction_description)
    ]
    val_mixed_c = [canonicalize(t) for t in val_mixed_noisy]
    # 2c. Pure 100% Typo noise validation
    r_typo = random.Random(999)
    val_typo_raw = [apply_single_noise(t, r_typo, "typo") for t in val_df.transaction_description]
    val_typo_c = [canonicalize(t) for t in val_typo_raw]

    # Feature extractors
    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
    Xtr_char = tfidf_char.fit_transform(train_c)
    Xva_clean_char = tfidf_char.transform(val_clean_c)
    Xva_mixed_char = tfidf_char.transform(val_mixed_c)
    Xva_typo_char  = tfidf_char.transform(val_typo_c)

    tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True)
    Xtr_word = tfidf_word.fit_transform(train_c)
    Xva_clean_word = tfidf_word.transform(val_clean_c)
    Xva_mixed_word = tfidf_word.transform(val_mixed_c)
    Xva_typo_word  = tfidf_word.transform(val_typo_c)

    scaler = MaxAbsScaler()
    Xtr_hc = csr_matrix(scaler.fit_transform(extract_handcrafted_features(train_c)))
    Xva_clean_hc = csr_matrix(scaler.transform(extract_handcrafted_features(val_clean_c)))
    Xva_mixed_hc = csr_matrix(scaler.transform(extract_handcrafted_features(val_mixed_c)))
    Xva_typo_hc  = csr_matrix(scaler.transform(extract_handcrafted_features(val_typo_c)))

    # Combinations:
    # B: Char only
    # B+Word: Char + Word
    # B+HC: Char + Handcrafted
    # C: Char + Word + Handcrafted
    Xtr_c = hstack([Xtr_char, Xtr_word, Xtr_hc]).tocsr()
    Xva_clean_c = hstack([Xva_clean_char, Xva_clean_word, Xva_clean_hc]).tocsr()
    Xva_mixed_c = hstack([Xva_mixed_char, Xva_mixed_word, Xva_mixed_hc]).tocsr()
    Xva_typo_c  = hstack([Xva_typo_char, Xva_typo_word, Xva_typo_hc]).tocsr()

    Xtr_char_word = hstack([Xtr_char, Xtr_word]).tocsr()
    Xva_clean_cw = hstack([Xva_clean_char, Xva_clean_word]).tocsr()
    Xva_mixed_cw = hstack([Xva_mixed_char, Xva_mixed_word]).tocsr()
    Xva_typo_cw  = hstack([Xva_typo_char, Xva_typo_word]).tocsr()

    # Fit models
    clf_b = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_char, y_tr)
    clf_cw = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_char_word, y_tr)
    clf_c = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_c, y_tr)
    clf_word_only = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_word, y_tr)

    print("\n--- Diagnostic Evaluation on Validation Splits ---")
    results = [
        {
            "Model_Variant": "Variant B (Char TF-IDF 20k)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_b.predict(Xva_clean_char)),
            "Mixed_Noisy_Val_Acc": accuracy_score(y_va, clf_b.predict(Xva_mixed_char)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_b.predict(Xva_typo_char)),
        },
        {
            "Model_Variant": "Word TF-IDF only (5k)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_word_only.predict(Xva_clean_word)),
            "Mixed_Noisy_Val_Acc": accuracy_score(y_va, clf_word_only.predict(Xva_mixed_word)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_word_only.predict(Xva_typo_word)),
        },
        {
            "Model_Variant": "Char + Word TF-IDF",
            "Clean_Val_Acc": accuracy_score(y_va, clf_cw.predict(Xva_clean_cw)),
            "Mixed_Noisy_Val_Acc": accuracy_score(y_va, clf_cw.predict(Xva_mixed_cw)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_cw.predict(Xva_typo_cw)),
        },
        {
            "Model_Variant": "Variant C (Char + Word + Handcrafted)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_c.predict(Xva_clean_c)),
            "Mixed_Noisy_Val_Acc": accuracy_score(y_va, clf_c.predict(Xva_mixed_c)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_c.predict(Xva_typo_c)),
        }
    ]
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Error breakdown on pure typo validation
    pred_b_typo = clf_b.predict(Xva_typo_char)
    pred_c_typo = clf_c.predict(Xva_typo_c)
    pred_w_typo = clf_word_only.predict(Xva_typo_word)

    # Why does Variant C lose to Variant B on typo?
    b_correct_c_wrong = [i for i in range(len(val_df)) if pred_b_typo[i] == y_va[i] and pred_c_typo[i] != y_va[i]]
    c_correct_b_wrong = [i for i in range(len(val_df)) if pred_c_typo[i] == y_va[i] and pred_b_typo[i] != y_va[i]]
    print(f"\nTypo Error Differential on Validation (N={len(val_df):,}):")
    print(f"  Variant B Correct, but Variant C Wrong: {len(b_correct_c_wrong)} ({len(b_correct_c_wrong)/len(val_df)*100:.2f}%)")
    print(f"  Variant C Correct, but Variant B Wrong: {len(c_correct_b_wrong)} ({len(c_correct_b_wrong)/len(val_df)*100:.2f}%)")

    print("\nConcrete Validation Examples where Variant B withstands typo but Variant C fails:")
    for idx in b_correct_c_wrong[:8]:
        orig = val_df.transaction_description.iloc[idx]
        typo_s = val_typo_raw[idx]
        true_c = classes[y_va[idx]]
        pb = classes[pred_b_typo[idx]]
        pc = classes[pred_c_typo[idx]]
        pw = classes[pred_w_typo[idx]]
        print(f"  Orig: '{orig}' -> Typo: '{typo_s}'")
        print(f"    True: {true_c} | Char Pred: {pb} [OK] | Word Pred: {pw} | Variant C Pred: {pc} [FAIL]")

    # -------------------------------------------------------------------------
    # STEP 4: CONTROLLED IMPROVEMENTS EVALUATED ON VALIDATION ONLY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: CONTROLLED CANDIDATE IMPROVEMENTS (TUNED ON VALIDATION ONLY)")
    print("=" * 80)

    # Candidate 1: Pure Character Extended n-grams (2-6 n-grams, 30,000 features, no word vectorizer)
    tfidf_char_ext = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3, max_features=30000, sublinear_tf=True, lowercase=False)
    Xtr_cext = tfidf_char_ext.fit_transform(train_c)
    Xva_clean_cext = tfidf_char_ext.transform(val_clean_c)
    Xva_mixed_cext = tfidf_char_ext.transform(val_mixed_c)
    Xva_typo_cext  = tfidf_char_ext.transform(val_typo_c)
    clf_cand1 = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_cext, y_tr)

    # Candidate 2: Char (20k) + Word (5k) with stronger L2 Regularization (C=1.0 instead of C=10)
    # Stronger regularization prevents model from over-indexing on exact word matches
    clf_cand2 = LogisticRegression(C=1.0, max_iter=300, random_state=42).fit(Xtr_char_word, y_tr)

    # Candidate 3: Char (20k) + Word (5k) with Feature Weight Balancing (Scaling Word features by 0.5)
    Xtr_scaled_word = hstack([Xtr_char, Xtr_word.multiply(0.5)]).tocsr()
    Xva_clean_sw = hstack([Xva_clean_char, Xva_clean_word.multiply(0.5)]).tocsr()
    Xva_mixed_sw = hstack([Xva_mixed_char, Xva_mixed_word.multiply(0.5)]).tocsr()
    Xva_typo_sw  = hstack([Xva_typo_char, Xva_typo_word.multiply(0.5)]).tocsr()
    clf_cand3 = LogisticRegression(C=10, max_iter=300, random_state=42).fit(Xtr_scaled_word, y_tr)

    candidates = [
        {
            "Candidate": "Baseline Variant B (Char TF-IDF 2-5, 20k, C=10)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_b.predict(Xva_clean_char)),
            "Mixed_Val_Acc": accuracy_score(y_va, clf_b.predict(Xva_mixed_char)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_b.predict(Xva_typo_char)),
            "Strategy": "Pure character n-grams"
        },
        {
            "Candidate": "Baseline Variant C (Char 20k + Word 5k + Handcrafted, C=10)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_c.predict(Xva_clean_c)),
            "Mixed_Val_Acc": accuracy_score(y_va, clf_c.predict(Xva_mixed_c)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_c.predict(Xva_typo_c)),
            "Strategy": "Stack char + word + dense"
        },
        {
            "Candidate": "Candidate 1: Extended Char TF-IDF (2-6 n-grams, 30k, C=10)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_cand1.predict(Xva_clean_cext)),
            "Mixed_Val_Acc": accuracy_score(y_va, clf_cand1.predict(Xva_mixed_cext)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_cand1.predict(Xva_typo_cext)),
            "Strategy": "Deep character subwords only"
        },
        {
            "Candidate": "Candidate 2: Char (20k) + Word (5k) with C=1.0 Regularization",
            "Clean_Val_Acc": accuracy_score(y_va, clf_cand2.predict(Xva_clean_cw)),
            "Mixed_Val_Acc": accuracy_score(y_va, clf_cand2.predict(Xva_mixed_cw)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_cand2.predict(Xva_typo_cw)),
            "Strategy": "Stronger L2 penalty prevents word overfitting"
        },
        {
            "Candidate": "Candidate 3: Char (20k) + Downweighted Word (0.5x, C=10)",
            "Clean_Val_Acc": accuracy_score(y_va, clf_cand3.predict(Xva_clean_sw)),
            "Mixed_Val_Acc": accuracy_score(y_va, clf_cand3.predict(Xva_mixed_sw)),
            "Pure_Typo_Val_Acc": accuracy_score(y_va, clf_cand3.predict(Xva_typo_sw)),
            "Strategy": "Balanced char-first hybrid"
        }
    ]
    cand_df = pd.DataFrame(candidates)
    print(cand_df.to_string(index=False))

    cand_df.to_csv(Path("results") / "canonicalization_typo_candidates_validation.csv", index=False)
    print("\nSaved validation candidates report to results/canonicalization_typo_candidates_validation.csv")


if __name__ == "__main__":
    main()
