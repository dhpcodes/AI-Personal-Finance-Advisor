"""
Train MiniLM Sentence Embeddings + Logistic Regression Classifier.
Script: scripts/train_minilm_lr.py

Strict Constraints:
- Fits on training split (mitulshah train + indian_merchants_train_seed.csv).
- Tunes ONLY regularization parameter C on validation split.
- Uses identical canonicalized preprocessing (TextPreprocessor).
- Caches embeddings in data/processed/*.npy for fast reruns.
- Evaluates on clean test, noisy test, typo test, and held-out Indian bank statements (N=105).
- Appends row to results/model_comparison.csv.
"""

import sys
import os
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import TextPreprocessor
from src.model_comparison_suite import with_rent_rule

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL_PATH = MODELS_DIR / "minilm_lr_transaction_classifier.joblib"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"
OWN_TEST_PATH = DATA_DIR / "own_test_indian.csv"

TEST_KINDS = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]

def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

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

def get_or_create_embeddings(texts: list, cache_path: Path, encoder: SentenceTransformer) -> np.ndarray:
    if cache_path.exists():
        print(f"  [CACHE HIT] Loading {cache_path.name} ({len(texts):,} vectors)...")
        return np.load(cache_path)
    print(f"  [CACHE MISS] Encoding {len(texts):,} texts with MiniLM...")
    t0 = time.time()
    embs = encoder.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    dur = time.time() - t0
    print(f"  Encoding completed in {dur:.1f}s ({len(texts)/max(dur, 0.001):.1f} texts/s). Saving to {cache_path.name}...")
    np.save(cache_path, embs)
    return embs

def main():
    print("=" * 70)
    print("TRAINING MINILM EMBEDDINGS + LOGISTIC REGRESSION CLASSIFIER")
    print("=" * 70)

    # 1. Load data
    print("\n[Step 1] Loading splits and Indian merchant seed...")
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)
    df_own = pd.read_csv(OWN_TEST_PATH)

    train_df = pd.concat([
        train_base[["transaction_description", "category"]],
        df_seed[["transaction_description", "category"]]
    ], ignore_index=True)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} | Indian: {len(df_own):,}")

    # 2. Preprocess text
    print("\n[Step 2] Preprocessing descriptions (canonicalize + clean_text)...")
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in train_df.transaction_description]
    val_clean = [preprocessor.clean_text(t) for t in val_df.transaction_description]
    test_clean = [preprocessor.clean_text(t) for t in test_df.transaction_description]
    own_clean = [preprocessor.clean_text(t) for t in df_own.transaction_description]

    # Noisy test set
    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
    test_noisy_descs = [apply_single_noise(t, r_te, nt) for t, nt in zip(test_df.transaction_description, test_noise_types)]
    test_noisy_clean = [preprocessor.clean_text(t) for t in test_noisy_descs]

    test_df["noise_type"] = test_noise_types
    test_df["noisy_description"] = test_noisy_descs

    # Labels
    le = LabelEncoder().fit(train_df.category)
    classes = list(le.classes_)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)
    y_te = le.transform(test_df.category)

    # 3. Embeddings
    print("\n[Step 3] Loading sentence-transformers/all-MiniLM-L6-v2...")
    model_name = "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model_name)
    encoder.max_seq_length = 32

    t0_embed = time.time()
    X_tr = get_or_create_embeddings(train_clean, CACHE_DIR / "minilm_train.npy", encoder)
    X_va = get_or_create_embeddings(val_clean, CACHE_DIR / "minilm_val.npy", encoder)
    X_te_clean = get_or_create_embeddings(test_clean, CACHE_DIR / "minilm_test_clean.npy", encoder)
    X_te_noisy = get_or_create_embeddings(test_noisy_clean, CACHE_DIR / "minilm_test_noisy.npy", encoder)
    X_own = get_or_create_embeddings(own_clean, CACHE_DIR / "minilm_own_indian.npy", encoder)
    embed_time = time.time() - t0_embed

    # 4. Tune C on Validation Split
    print("\n[Step 4] Tuning Logistic Regression C on validation split...")
    candidate_c = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_c = 1.0
    best_val_f1 = -1.0
    best_val_acc = -1.0

    for c_val in candidate_c:
        clf_try = LogisticRegression(C=c_val, max_iter=500, random_state=42, n_jobs=-1)
        clf_try.fit(X_tr, y_tr)
        p_va = clf_try.predict(X_va)
        acc_v = accuracy_score(y_va, p_va)
        f1_v = f1_score(y_va, p_va, average="macro", zero_division=0)
        print(f"  C={c_val:4.1f} -> Val Acc: {acc_v:.4f}, Val Macro-F1: {f1_v:.4f}")
        if f1_v > best_val_f1:
            best_val_f1 = f1_v
            best_val_acc = acc_v
            best_c = c_val

    print(f"  Best C selected: {best_c} (Val Macro-F1: {best_val_f1:.4f})")

    # 5. Fit Final Classifier
    print(f"\n[Step 5] Fitting final Logistic Regression with C={best_c}...")
    t0_fit = time.time()
    final_clf = LogisticRegression(C=best_c, max_iter=500, random_state=42, n_jobs=-1)
    final_clf.fit(X_tr, y_tr)
    fit_time = time.time() - t0_fit
    print(f"  Model fitted in {fit_time:.2f}s.")

    # 6. Evaluation
    print("\n[Step 6] Evaluating on untouched test sets...")
    # Clean Test
    preds_clean = final_clf.predict(X_te_clean)
    clean_acc = accuracy_score(y_te, preds_clean)
    clean_macro_f1 = f1_score(y_te, preds_clean, average="macro", zero_division=0)
    clean_weighted_f1 = f1_score(y_te, preds_clean, average="weighted", zero_division=0)

    # Noisy Test
    preds_noisy = final_clf.predict(X_te_noisy)
    noisy_acc = accuracy_score(y_te, preds_noisy)
    noisy_macro_f1 = f1_score(y_te, preds_noisy, average="macro", zero_division=0)
    noisy_weighted_f1 = f1_score(y_te, preds_noisy, average="weighted", zero_division=0)

    # Typo subset
    test_df["pred_noisy"] = preds_noisy
    typo_mask = test_df.noise_type == "typo"
    typo_acc = accuracy_score(test_df.loc[typo_mask, "category"], [classes[p] for p in test_df.loc[typo_mask, "pred_noisy"]])

    # Real Indian Statements
    preds_own_idx = final_clf.predict(X_own)
    preds_own = [classes[i] for i in preds_own_idx]
    own_model_acc = accuracy_score(df_own.category, preds_own)

    # Model + Rent Rule
    own_rule_preds = with_rent_rule(df_own.transaction_description, pd.Series(preds_own))
    own_rule_acc = accuracy_score(df_own.category, own_rule_preds)
    own_rule_f1 = f1_score(df_own.category, own_rule_preds, average="macro", zero_division=0)

    # Latency: single item
    latencies = []
    sample_texts = test_clean[:200]
    for s in sample_texts[:20]:  # warmup
        emb = encoder.encode([s], show_progress_bar=False, normalize_embeddings=True)
        _ = final_clf.predict(emb)
    for s in sample_texts:
        t0 = time.perf_counter()
        emb = encoder.encode([s], show_progress_bar=False, normalize_embeddings=True)
        _ = final_clf.predict(emb)
        latencies.append(time.perf_counter() - t0)
    single_latency_ms = float(np.mean(latencies) * 1000.0)

    print(f"  Clean Test Acc:   {clean_acc:.4f} ({clean_acc*100:.2f}%) | Macro-F1: {clean_macro_f1:.4f}")
    print(f"  Noisy Test Acc:   {noisy_acc:.4f} ({noisy_acc*100:.2f}%) | Macro-F1: {noisy_macro_f1:.4f}")
    print(f"  Typo Noise Acc:   {typo_acc:.4f} ({typo_acc*100:.2f}%)")
    print(f"  Indian Model-Only:{own_model_acc:.4f} ({own_model_acc*100:.2f}%)")
    print(f"  Indian Model+Rent:{own_rule_acc:.4f} ({own_rule_acc*100:.2f}%) | Macro-F1: {own_rule_f1:.4f}")
    print(f"  Single-item Latency (encode+predict): {single_latency_ms:.2f} ms/txn")

    # 7. Save Model Artifact
    print(f"\n[Step 7] Saving model artifact to {OUTPUT_MODEL_PATH}...")
    artifact = {
        "version": "1.0.0-minilm_lr",
        "model_name": "MiniLM-L6-v2 + Logistic Regression",
        "encoder_name": model_name,
        "feature_dim": 384,
        "best_c": best_c,
        "model": final_clf,
        "classes": classes,
        "label_encoder": le,
        "metrics": {
            "val_accuracy": round(best_val_acc, 4),
            "val_macro_f1": round(best_val_f1, 4),
            "fit_time_sec": round(fit_time, 2),
            "embed_time_sec": round(embed_time, 2),
            "single_latency_ms": round(single_latency_ms, 2)
        }
    }
    joblib.dump(artifact, OUTPUT_MODEL_PATH)
    art_hash = get_file_sha256(OUTPUT_MODEL_PATH)
    art_size_mb = OUTPUT_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  Artifact saved. Size: {art_size_mb:.2f} MB | SHA256: {art_hash}")

    # 8. Append to results/model_comparison.csv
    minilm_row = {
        "Model": "MiniLM_LogReg",
        "Validation_Accuracy": round(best_val_acc, 4),
        "Validation_Macro_F1": round(best_val_f1, 4),
        "Clean_Test_Accuracy": round(clean_acc, 4),
        "Clean_Test_Macro_F1": round(clean_macro_f1, 4),
        "Clean_Test_Weighted_F1": round(clean_weighted_f1, 4),
        "Noisy_Test_Accuracy": round(noisy_acc, 4),
        "Noisy_Test_Macro_F1": round(noisy_macro_f1, 4),
        "Noisy_Test_Weighted_F1": round(noisy_weighted_f1, 4),
        "Typo_Accuracy": round(typo_acc, 4),
        "Indian_Statement_Model_Acc": round(own_model_acc, 4),
        "Indian_Statement_Model_Plus_RentRule_Acc": round(own_rule_acc, 4),
        "Indian_Statement_Macro_F1": round(own_rule_f1, 4),
        "Training_Time_Sec": round(fit_time, 2),
        "Inference_Latency_ms_per_txn": round(single_latency_ms, 3),
        "Artifact_Size_MB": round(art_size_mb, 2)
    }

    if COMPARISON_CSV.exists():
        df_comp = pd.read_csv(COMPARISON_CSV)
        df_comp = df_comp[df_comp["Model"] != "MiniLM_LogReg"]
        df_comp = pd.concat([df_comp, pd.DataFrame([minilm_row])], ignore_index=True)
        df_comp.to_csv(COMPARISON_CSV, index=False)
        print(f"  Appended MiniLM_LogReg metrics to {COMPARISON_CSV}.")

    print("\n[DONE] MiniLM + Logistic Regression training and evaluation complete.")

if __name__ == "__main__":
    main()
