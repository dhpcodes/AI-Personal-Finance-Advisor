"""Train XGBoost Transaction Classifier.

Strict safety constraints:
- Does NOT touch models/final_model.joblib or backups.
- Fits vectorizer and model exclusively on training data + indian_merchants_train_seed.csv.
- Evaluates on validation split for model tuning.
- Saves artifact to models/xgboost_transaction_classifier.joblib.
"""

import sys
import time
import hashlib
import joblib
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier

from src.preprocessing import TextPreprocessor

DATA_SPLIT_PATH = REPO_ROOT / "data" / "mitulshah_split.csv"
INDIAN_SEED_PATH = REPO_ROOT / "data" / "indian_merchants_train_seed.csv"
OWN_TEST_PATH = REPO_ROOT / "data" / "own_test_indian.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODEL_PATH = MODELS_DIR / "xgboost_transaction_classifier.joblib"


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("TRAINING XGBOOST TRANSACTION CLASSIFIER")
    print("=" * 70)

    # 1. Load data
    print("\n[Step 1] Loading dataset splits and Indian merchant seed...")
    df_split = pd.read_csv(DATA_SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    df_seed = pd.read_csv(INDIAN_SEED_PATH)

    # Strict leakage check
    df_own_check = pd.read_csv(OWN_TEST_PATH)
    overlap = set(df_seed["transaction_description"].str.strip().str.upper()).intersection(
        set(df_own_check["transaction_description"].str.strip().str.upper())
    )
    assert len(overlap) == 0, f"FATAL: Overlap detected with held-out test: {overlap}"
    print(f"  [LEAKAGE CHECK PASS] 0 overlap between Indian seed and held-out test.")

    # Combine base train + Indian seed strictly in train split
    train_df = pd.concat([
        train_base[["transaction_description", "category"]],
        df_seed[["transaction_description", "category"]]
    ], ignore_index=True)
    print(f"  Train samples: {len(train_df):,} (Base: {len(train_base):,}, Seed: {len(df_seed):,})")
    print(f"  Val samples:   {len(val_df):,}")

    # 2. Text preprocessing
    print("\n[Step 2] Preprocessing transaction text with TextPreprocessor (canonicalize -> clean_text)...")
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in train_df.transaction_description]
    val_clean = [preprocessor.clean_text(t) for t in val_df.transaction_description]

    # Label encoding
    le = LabelEncoder().fit(train_df.category)
    classes = list(le.classes_)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)

    # 3. Feature Extraction
    print("\n[Step 3] Fitting TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=3000)...")
    t0_vec = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=5,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    X_tr = vectorizer.fit_transform(train_clean)
    X_va = vectorizer.transform(val_clean)
    vec_time = time.time() - t0_vec
    print(f"  Vectorizer fitted in {vec_time:.2f}s | Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  Feature matrix shape: Train {X_tr.shape}, Val {X_va.shape}")

    # 4. Model Training
    print("\n[Step 4] Training XGBoost Classifier (80 trees, lr=0.20, max_depth=6, colsample=0.8)...")
    t0_fit = time.time()
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=80,
        learning_rate=0.20,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss"
    )
    xgb_model.fit(X_tr, y_tr)
    fit_time = time.time() - t0_fit
    print(f"  Model trained in {fit_time:.2f}s.")

    # 5. Validation Evaluation
    print("\n[Step 5] Evaluating on held-out validation split...")
    val_preds = xgb_model.predict(X_va)
    val_acc = float(accuracy_score(y_va, val_preds))
    val_f1 = float(f1_score(y_va, val_preds, average="macro"))
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Validation Macro-F1: {val_f1:.4f}")

    # 6. Save Artifact
    print(f"\n[Step 6] Saving model artifact to {OUTPUT_MODEL_PATH}...")
    artifact = {
        "version": "1.0.0-xgboost",
        "model_name": "XGBoost Multiclass Classifier (char_wb 2-4)",
        "vectorizer": vectorizer,
        "label_encoder": le,
        "model": xgb_model,
        "classes": classes,
        "metrics": {
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "train_samples": len(train_df),
            "fit_time_sec": round(fit_time, 2),
            "vec_time_sec": round(vec_time, 2)
        },
        "preprocessing_metadata": {
            "pipeline": "canonicalize -> clean_text (char_wb 2-4)",
            "vectorizer_params": vectorizer.get_params(),
            "indian_seed_samples": len(df_seed),
            "hyperparameters": {
                "n_estimators": 80,
                "learning_rate": 0.20,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist"
            }
        }
    }
    joblib.dump(artifact, OUTPUT_MODEL_PATH)
    file_size_mb = OUTPUT_MODEL_PATH.stat().st_size / (1024 * 1024)
    file_hash = get_file_sha256(OUTPUT_MODEL_PATH)
    print(f"  Artifact saved successfully.")
    print(f"  Size:   {file_size_mb:.2f} MB")
    print(f"  SHA256: {file_hash}")
    print("\n[DONE] XGBoost training pipeline complete.")


if __name__ == "__main__":
    main()
