"""
Model 6: Fine-tuned ModernBERT / DistilBERT for Transaction Categorization
Reproducible Colab GPU Training Workflow

Requirements (Google Colab with NVIDIA T4/A100 GPU):
    !pip install -q transformers datasets evaluate accelerate torch scikit-learn

Execution:
    python notebooks/05_modernbert_training.py --model_name answerdotai/ModernBERT-base
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "macro_f1": f1}

def train_transformer(model_name="answerdotai/ModernBERT-base", epochs=3, batch_size=32):
    split_path = "data/mitulshah_split.csv"
    if not os.path.exists(split_path):
        print("Dataset split file missing.")
        return

    df = pd.read_csv(split_path)
    categories = sorted(df["category"].unique())
    cat2idx = {c: i for i, c in enumerate(categories)}
    idx2cat = {i: c for i, c in enumerate(categories)}

    df["label"] = df["category"].map(cat2idx)
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(batch):
        return tokenizer(batch["transaction_description"], truncation=True, max_length=64)

    train_ds = Dataset.from_pandas(train_df[["transaction_description", "label"]]).map(tokenize_func, batched=True)
    val_ds = Dataset.from_pandas(val_df[["transaction_description", "label"]]).map(tokenize_func, batched=True)
    test_ds = Dataset.from_pandas(test_df[["transaction_description", "label"]]).map(tokenize_func, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label=idx2cat,
        label2id=cat2idx
    )

    training_args = TrainingArguments(
        output_dir=f"./checkpoints_{model_name.replace('/', '_')}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Beginning fine-tuning...")
    trainer.train()

    print("Evaluating on unseen merchant test set...")
    test_results = trainer.evaluate(test_ds)
    print("\n--- Model 6 Test Set Performance ---")
    print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Test Macro-F1: {test_results.get('eval_macro_f1', 'N/A'):.4f}")

if __name__ == "__main__":
    train_transformer()
