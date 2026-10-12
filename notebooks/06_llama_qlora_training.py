"""
Model 7: Llama 3.2 1B Instruct with QLoRA for 10-Class Transaction Categorization
Reproducible Colab GPU Workflow

Requirements (Google Colab with NVIDIA A10G / A100 GPU):
    !pip install -q transformers datasets peft bitsandbytes trl accelerate torch

Execution:
    python notebooks/06_llama_qlora_training.py
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

SYSTEM_PROMPT = """You are an expert financial transaction classification assistant.
Given a raw transaction description, classify it into exactly one of these 10 categories:
1. Charity & Donations
2. Entertainment & Recreation
3. Financial Services
4. Food & Dining
5. Government & Legal
6. Healthcare & Medical
7. Income
8. Shopping & Retail
9. Transportation
10. Utilities & Services

Respond ONLY with the category name."""

def format_instruction(sample):
    return f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\nTransaction: {sample['transaction_description']}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{sample['category']}<|eot_id|>"

def train_llama_qlora():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    split_path = "data/mitulshah_split.csv"

    if not os.path.exists(split_path):
        print("Data split missing.")
        return

    df = pd.read_csv(split_path)
    train_df = df[df.split == "train"].copy()
    val_df = df[df.split == "val"].copy()

    train_df["text"] = train_df.apply(format_instruction, axis=1)
    val_df["text"] = val_df.apply(format_instruction, axis=1)

    train_ds = Dataset.from_pandas(train_df[["text"]])
    val_ds = Dataset.from_pandas(val_df[["text"]])

    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./checkpoints_llama_qlora",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=128,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config
    )

    print("Starting QLoRA instruction tuning...")
    trainer.train()
    print("QLoRA training complete.")

if __name__ == "__main__":
    train_llama_qlora()
