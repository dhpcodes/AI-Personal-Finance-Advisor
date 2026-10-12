"""
Model 5: Sentence Embeddings (all-MiniLM-L6-v2) + Logistic Regression
Reproducible Workflow Script

Requirements:
    pip install sentence-transformers scikit-learn pandas numpy

Usage:
    python notebooks/04b_sentence_embeddings.py
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer

def run_sentence_embeddings_experiment():
    split_path = "data/mitulshah_split.csv"
    if not os.path.exists(split_path):
        print(f"Error: {split_path} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(split_path)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]

    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model_name = "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model_name)

    print("Encoding training descriptions (384-dim embeddings)...")
    t0 = time.time()
    X_train = encoder.encode(train["transaction_description"].tolist(), batch_size=64, show_progress_bar=True)
    t_train_enc = time.time() - t0
    print(f"Training encoding complete in {t_train_enc:.1f}s")

    X_test = encoder.encode(test["transaction_description"].tolist(), batch_size=64, show_progress_bar=True)

    le = LabelEncoder()
    y_train = le.fit_transform(train["category"])
    y_test = le.transform(test["category"])

    print("Fitting Logistic Regression on sentence embeddings...")
    clf = LogisticRegression(C=5.0, max_iter=500, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n--- Model 5 Results ({model_name} + LogReg) ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le.classes_))

if __name__ == "__main__":
    run_sentence_embeddings_experiment()
