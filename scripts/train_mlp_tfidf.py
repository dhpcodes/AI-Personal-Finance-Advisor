"""Train Non-Linear MLP Classifier on TF-IDF Features (Task 1).

Specifications:
- TF-IDF char_wb(2, 4), 3,000 features (fit on train only).
- MLPClassifier: one hidden layer of 256 units, ReLU, early_stopping=True, random_state=42.
- Tunes alpha in [1e-4, 1e-3, 1e-2] on the validation set only.
- Saves model artifact to models/mlp_tfidf_classifier.joblib.
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import TextPreprocessor

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
SPLIT_PATH = DATA_DIR / "mitulshah_split.csv"
SEED_PATH = DATA_DIR / "indian_merchants_train_seed.csv"
OUTPUT_MODEL_PATH = MODELS_DIR / "mlp_tfidf_classifier.joblib"

def main():
    print("=" * 70)
    print("TASK 1: TRAINING NON-LINEAR MLP ON TF-IDF")
    print("=" * 70)

    # 1. Load data
    df_split = pd.read_csv(SPLIT_PATH)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(SEED_PATH)

    df_train = pd.concat([train_base, df_seed], ignore_index=True)
    print(f"Train samples: {len(df_train):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")

    # 2. Canonicalized Preprocessing
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in df_train["transaction_description"]]
    val_clean = [preprocessor.clean_text(t) for t in val_df["transaction_description"]]
    test_clean = [preprocessor.clean_text(t) for t in test_df["transaction_description"]]

    le = LabelEncoder().fit(df_train["category"])
    classes = list(le.classes_)
    y_tr = le.transform(df_train["category"])
    y_va = le.transform(val_df["category"])
    y_te = le.transform(test_df["category"])

    # 3. TF-IDF feature extraction (3,000 features, char_wb 2-4)
    print("\n[Step 2] Fitting TF-IDF Vectorizer (char_wb 2-4, 3000 max_features)...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=5,
        max_features=3000,
        sublinear_tf=True,
        lowercase=True
    )
    t0 = time.time()
    X_tr = vectorizer.fit_transform(train_clean)
    X_va = vectorizer.transform(val_clean)
    X_te = vectorizer.transform(test_clean)
    print(f"TF-IDF fitted in {time.time()-t0:.2f}s | Shape: {X_tr.shape}")

    # 4. Tuning alpha on Validation Set
    print("\n[Step 3] Tuning alpha on Validation Set...")
    alpha_candidates = [1e-4, 1e-3, 1e-2]
    best_alpha = None
    best_val_acc = -1.0
    best_model = None
    best_fit_time = 0.0

    for alpha in alpha_candidates:
        print(f"\n  Evaluating alpha={alpha}...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=alpha,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
            batch_size=256
        )
        t_fit_start = time.time()
        mlp.fit(X_tr, y_tr)
        fit_time = time.time() - t_fit_start

        val_preds = mlp.predict(X_va)
        val_acc = accuracy_score(y_va, val_preds)
        val_f1 = f1_score(y_va, val_preds, average="macro")
        print(f"    Fit time: {fit_time:.1f}s | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | Val F1: {val_f1:.4f} | Epochs: {mlp.n_iter_}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_alpha = alpha
            best_model = mlp
            best_fit_time = fit_time

    print(f"\nSelected Best alpha={best_alpha} with Val Acc={best_val_acc:.4f}")

    # 5. Evaluate on Test Split
    test_preds = best_model.predict(X_te)
    test_acc = accuracy_score(y_te, test_preds)
    test_f1 = f1_score(y_te, test_preds, average="macro")
    print(f"Clean Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%) | Test F1: {test_f1:.4f}")

    # Train accuracy
    train_preds = best_model.predict(X_tr)
    train_acc = accuracy_score(y_tr, train_preds)
    print(f"Train Acc:      {train_acc:.4f} ({train_acc*100:.2f}%)")

    # 6. Save Artifact
    artifact = {
        "version": "1.0.0-mlp_tfidf",
        "model_name": "MLP (256 ReLU, TF-IDF char_wb 2-4)",
        "vectorizer": vectorizer,
        "label_encoder": le,
        "model": best_model,
        "classes": classes,
        "model_type": "tfidf_mlp",
        "best_alpha": best_alpha,
        "dimensionality_reduction": "None (Direct sparse 3,000 features)",
        "metrics": {
            "train_accuracy": round(float(train_acc), 4),
            "val_accuracy": round(float(best_val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_f1), 4),
            "fit_time_sec": round(float(best_fit_time), 2)
        }
    }
    joblib.dump(artifact, OUTPUT_MODEL_PATH)
    print(f"\n[SAVED] Saved MLP model artifact to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
