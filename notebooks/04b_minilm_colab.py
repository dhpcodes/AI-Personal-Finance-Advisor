"""
Model 5: MiniLM-L6-v2 Embeddings + Logistic Regression on Google Colab
Script: notebooks/04b_minilm_colab.py

INSTRUCTIONS FOR USER TO RUN ON GOOGLE COLAB:
=============================================
1. Open Google Colab (https://colab.research.google.com), select Runtime -> Change runtime type -> T4 GPU (or CPU).
2. Create a 'data/' folder in Colab and upload:
   - data/mitulshah_split.csv
   - data/indian_merchants_train_seed.csv
   - data/own_test_indian.csv
3. Run:
   !pip install sentence-transformers scikit-learn pandas numpy
   !python 04b_minilm_colab.py
4. Download the generated model artifact:
   - models/minilm_lr_transaction_classifier.joblib
5. Place it in your local repository under 'models/'.
"""

import os
import sys
import time
import random
import re
import hashlib
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Canonicalization & Text Preprocessing
# ---------------------------------------------------------
def canonicalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip()
    t = re.sub(r'(?i)^UPI/[\d]+/', '', t)
    t = re.sub(r'(?i)@[a-zA-Z0-9\.\-_]+', '', t)
    t = re.sub(r'(?i)^POS\s+\d{4}X+\d{4}\s+', '', t)
    t = re.sub(r'(?i)^POS\s+', '', t)
    t = re.sub(r'(?i)^NEFT[-\s]+[\w\d]+[-\s]+', '', t)
    t = re.sub(r'(?i)^IMPS/[\d]+/', '', t)
    t = re.sub(r'(?i)/REF[\d]+', '', t)
    return t.strip()

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = canonicalize(text).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'[^\w\s\-\#\&]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

TEST_KINDS = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]

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

def with_rent_rule(descs, preds):
    rent_keywords = ["RENT", "LANDLORD", "PG RENT", "HOSTEL", "MAINTENANCE & RENT", "HOUSE LEASE ADVANCE", "FLAT 301"]
    out = []
    for d, p in zip(descs, preds):
        s = str(d).upper()
        if any(k in s for k in rent_keywords):
            out.append("Rent")
        else:
            out.append(p)
    return out

def main():
    print("=" * 70)
    print("MINILM + LOGISTIC REGRESSION TRAINING (COLAB / CPU)")
    print("=" * 70)

    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = models_dir / "minilm_lr_transaction_classifier.joblib"

    df_split = pd.read_csv(data_dir / "mitulshah_split.csv")
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(data_dir / "indian_merchants_train_seed.csv")
    df_own = pd.read_csv(data_dir / "own_test_indian.csv")

    train_df = pd.concat([
        train_base[["transaction_description", "category"]],
        df_seed[["transaction_description", "category"]]
    ], ignore_index=True)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,} | Test: {len(test_df):,}")

    train_clean = [clean_text(t) for t in train_df.transaction_description]
    val_clean = [clean_text(t) for t in val_df.transaction_description]
    test_clean = [clean_text(t) for t in test_df.transaction_description]
    own_clean = [clean_text(t) for t in df_own.transaction_description]

    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
    test_noisy_descs = [apply_single_noise(t, r_te, nt) for t, nt in zip(test_df.transaction_description, test_noise_types)]
    test_noisy_clean = [clean_text(t) for t in test_noisy_descs]
    test_df["noise_type"] = test_noise_types

    le = LabelEncoder().fit(train_df.category)
    classes = list(le.classes_)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)
    y_te = le.transform(test_df.category)

    print("\nEncoding with all-MiniLM-L6-v2...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    encoder.max_seq_length = 32

    t0_enc = time.time()
    X_tr = encoder.encode(train_clean, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    X_va = encoder.encode(val_clean, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    X_te_clean = encoder.encode(test_clean, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    X_te_noisy = encoder.encode(test_noisy_clean, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    X_own = encoder.encode(own_clean, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    enc_time = time.time() - t0_enc
    print(f"Encoding finished in {enc_time:.1f}s.")

    print("\nTuning C on validation split...")
    candidate_c = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_c, best_f1, best_acc = 1.0, -1.0, -1.0
    for c_val in candidate_c:
        clf = LogisticRegression(C=c_val, max_iter=500, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_va, preds)
        print(f"  C={c_val:4.1f} -> Val Acc: {acc:.4f}, Val Macro-F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_acc, best_c = f1, acc, c_val

    print(f"Selected C={best_c} (Val Macro-F1: {best_f1:.4f})")

    final_clf = LogisticRegression(C=best_c, max_iter=500, random_state=42, n_jobs=-1)
    t0_fit = time.time()
    final_clf.fit(X_tr, y_tr)
    fit_time = time.time() - t0_fit

    # Evaluation
    preds_clean = final_clf.predict(X_te_clean)
    clean_acc = accuracy_score(y_te, preds_clean)
    clean_macro_f1 = f1_score(y_te, preds_clean, average="macro", zero_division=0)

    preds_noisy = final_clf.predict(X_te_noisy)
    noisy_acc = accuracy_score(y_te, preds_noisy)
    noisy_macro_f1 = f1_score(y_te, preds_noisy, average="macro", zero_division=0)

    test_df["pred_noisy"] = preds_noisy
    typo_mask = test_df.noise_type == "typo"
    typo_acc = accuracy_score(test_df.loc[typo_mask, "category"], [classes[p] for p in test_df.loc[typo_mask, "pred_noisy"]])

    preds_own_idx = final_clf.predict(X_own)
    preds_own = [classes[i] for i in preds_own_idx]
    own_model_acc = accuracy_score(df_own.category, preds_own)
    own_rule_preds = with_rent_rule(df_own.transaction_description, preds_own)
    own_rule_acc = accuracy_score(df_own.category, own_rule_preds)
    own_rule_f1 = f1_score(df_own.category, own_rule_preds, average="macro", zero_division=0)

    print("\n--- RESULTS ---")
    print(f"Clean Test Acc:   {clean_acc:.4f} | Macro-F1: {clean_macro_f1:.4f}")
    print(f"Noisy Test Acc:   {noisy_acc:.4f} | Macro-F1: {noisy_macro_f1:.4f}")
    print(f"Typo Noise Acc:   {typo_acc:.4f}")
    print(f"Indian Model-Only:{own_model_acc:.4f}")
    print(f"Indian Model+Rent:{own_rule_acc:.4f} | Macro-F1: {own_rule_f1:.4f}")

    # Save
    artifact = {
        "version": "1.0.0-minilm_lr",
        "model_name": "MiniLM-L6-v2 + Logistic Regression",
        "encoder_name": "all-MiniLM-L6-v2",
        "feature_dim": 384,
        "best_c": best_c,
        "model": final_clf,
        "classes": classes,
        "label_encoder": le,
        "metrics": {
            "val_accuracy": round(best_acc, 4),
            "val_macro_f1": round(best_f1, 4),
            "fit_time_sec": round(fit_time, 2),
            "embed_time_sec": round(enc_time, 2)
        }
    }
    joblib.dump(artifact, out_model_path)
    print(f"\nArtifact saved to {out_model_path}")

if __name__ == "__main__":
    main()
