"""
Model 6: DistilBERT Fine-Tuning on Google Colab (Free T4 GPU)
Script: notebooks/05_distilbert_colab.py

INSTRUCTIONS FOR USER TO RUN ON GOOGLE COLAB:
=============================================
1. Open Google Colab (https://colab.research.google.com), select Runtime -> Change runtime type -> T4 GPU.
2. In the Colab file browser, create a 'data/' folder and upload:
   - data/mitulshah_split.csv
   - data/indian_merchants_train_seed.csv
   - data/own_test_indian.csv
3. Run this script:
   !pip install torch transformers datasets scikit-learn pandas numpy accelerate
   !python 05_distilbert_colab.py
4. Download the generated prediction CSVs from the 'results_distilbert/' folder:
   - distilbert_preds_clean_test.csv
   - distilbert_preds_noisy_test.csv
   - distilbert_preds_typo_test.csv
   - distilbert_preds_own_test_indian.csv
5. Place them in your local repository under 'results/distilbert/' and run:
   python scripts/eval_distilbert_predictions.py
"""

import os
import re
import random
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# ---------------------------------------------------------
# 1. Canonicalization & Preprocessing (Self-contained for Colab)
# ---------------------------------------------------------
def canonicalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip()
    # Strip UPI wrappers
    t = re.sub(r'(?i)^UPI/[\d]+/', '', t)
    t = re.sub(r'(?i)@[a-zA-Z0-9\.\-_]+', '', t)
    # Strip POS card masks
    t = re.sub(r'(?i)^POS\s+\d{4}X+\d{4}\s+', '', t)
    t = re.sub(r'(?i)^POS\s+', '', t)
    # Strip NEFT / IMPS prefixes
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

def main():
    print("=" * 70)
    print("DISTILBERT FINE-TUNING PIPELINE (Colab T4)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute device: {device}")
    if device != "cuda":
        print("WARNING: Running without GPU. Colab GPU acceleration recommended.")

    # 1. Load data
    data_dir = Path("data")
    split_path = data_dir / "mitulshah_split.csv"
    seed_path = data_dir / "indian_merchants_train_seed.csv"
    own_path = data_dir / "own_test_indian.csv"

    out_dir = Path("results_distilbert")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_split = pd.read_csv(split_path)
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv(seed_path)
    df_own = pd.read_csv(own_path)

    # Combine training data
    train_df = pd.concat([
        train_base[["transaction_description", "category"]],
        df_seed[["transaction_description", "category"]]
    ], ignore_index=True)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,} | Test samples: {len(test_df):,}")

    # Encode labels
    le = LabelEncoder().fit(train_df.category)
    classes = list(le.classes_)
    num_classes = len(classes)
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}

    train_df["label"] = le.transform(train_df.category)
    val_df["label"] = le.transform(val_df.category)
    test_df["label"] = le.transform(test_df.category)

    # Preprocess text
    train_df["clean_text"] = [clean_text(t) for t in train_df.transaction_description]
    val_df["clean_text"] = [clean_text(t) for t in val_df.transaction_description]
    test_df["clean_text"] = [clean_text(t) for t in test_df.transaction_description]

    # Generate noisy and typo test sets
    r_te = random.Random(7)
    test_noise_types = [r_te.choice(TEST_KINDS) for _ in range(len(test_df))]
    test_noisy_descs = [apply_single_noise(t, r_te, nt) for t, nt in zip(test_df.transaction_description, test_noise_types)]
    test_df["noisy_description"] = test_noisy_descs
    test_df["noise_type"] = test_noise_types
    test_df["clean_noisy_text"] = [clean_text(t) for t in test_noisy_descs]

    # Tokenizer & Model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )

    def tokenize_fn(batch):
        return tokenizer(batch["clean_text"], truncation=True, max_length=32, padding="max_length")

    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)
    val_dataset = Dataset.from_pandas(val_df[["clean_text", "label"]]).map(tokenize_fn, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": macro_f1}

    training_args = TrainingArguments(
        output_dir="distilbert_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
        logging_steps=100,
        seed=42,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("\nStarting DistilBERT fine-tuning...")
    t0_fit = time.time()
    trainer.train()
    fit_time = time.time() - t0_fit
    print(f"Training completed in {fit_time:.2f}s ({fit_time/60:.1f} mins).")

    # Helper function for batch prediction
    def predict_texts(texts):
        enc = tokenizer(texts, truncation=True, max_length=32, padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        model.eval()
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)
        pred_labels = [classes[idx] for idx in pred_ids]
        confs = np.max(probs, axis=-1)
        return pred_labels, confs

    # 1. Clean Test Predictions
    print("\nPredicting Clean Test Set...")
    batch_size = 256
    clean_texts = test_df["clean_text"].tolist()
    all_preds, all_confs = [], []
    for i in range(0, len(clean_texts), batch_size):
        preds, confs = predict_texts(clean_texts[i:i+batch_size])
        all_preds.extend(preds)
        all_confs.extend(confs)
    df_clean_out = pd.DataFrame({
        "transaction_description": test_df.transaction_description,
        "true_category": test_df.category,
        "predicted_category": all_preds,
        "confidence": all_confs
    })
    df_clean_out.to_csv(out_dir / "distilbert_preds_clean_test.csv", index=False)

    # 2. Noisy Test Predictions
    print("Predicting Noisy Test Set...")
    noisy_texts = test_df["clean_noisy_text"].tolist()
    all_preds_n, all_confs_n = [], []
    for i in range(0, len(noisy_texts), batch_size):
        preds, confs = predict_texts(noisy_texts[i:i+batch_size])
        all_preds_n.extend(preds)
        all_confs_n.extend(confs)
    df_noisy_out = pd.DataFrame({
        "transaction_description": test_df.noisy_description,
        "noise_type": test_df.noise_type,
        "true_category": test_df.category,
        "predicted_category": all_preds_n,
        "confidence": all_confs_n
    })
    df_noisy_out.to_csv(out_dir / "distilbert_preds_noisy_test.csv", index=False)

    # 3. Typo Subset Predictions
    typo_mask = test_df.noise_type == "typo"
    df_typo_out = df_noisy_out[typo_mask].copy()
    df_typo_out.to_csv(out_dir / "distilbert_preds_typo_test.csv", index=False)

    # 4. Held-out Real Indian Statements
    print("Predicting Indian Bank Statements (N=105)...")
    own_clean = [clean_text(t) for t in df_own.transaction_description]
    preds_own, confs_own = predict_texts(own_clean)
    df_own_out = pd.DataFrame({
        "transaction_description": df_own.transaction_description,
        "true_category": df_own.category,
        "predicted_category": preds_own,
        "confidence": confs_own
    })
    df_own_out.to_csv(out_dir / "distilbert_preds_own_test_indian.csv", index=False)

    print("\n[DONE] All prediction CSVs saved to results_distilbert/.")

if __name__ == "__main__":
    main()
