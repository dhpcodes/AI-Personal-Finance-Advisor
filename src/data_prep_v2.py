import os
import re
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from datasets import Dataset

# Define paths
DATA_DIR = Path("data")
RAW_PARQUET = DATA_DIR / "raw" / "0000.parquet"
SAMPLE_100K_PATH = DATA_DIR / "mitulshah_sample_100k.csv"
SPLIT_CSV_PATH = DATA_DIR / "mitulshah_split.csv"

# Pre-compiled regex patterns for core_merchant normalization
BRACKETS_RE = re.compile(r"\(.*?\)|\[.*?\]")
TXN_REF_RE = re.compile(r"\b(txn\d+|ref\d+|#\d+|\d{4,})\b", re.I)
COUNTRY_SUFFIX_RE = re.compile(r"-\s*(canada|usa|india|uk|australia)\b", re.I)
STORE_NUM_RE = re.compile(r"#\s*\d+")

# Filler tokens to strip
FILLER_WORDS = [
    r"\bstore\b", r"\bbranch\b", r"\bcenter\b", r"\bcentre\b", r"\bmall\b",
    r"\bstrip mall\b", r"\bonline\b", r"\bresidential\b", r"\bbusiness district\b",
    r"\bexpress\b", r"\bsupercenter\b", r"\bsuperstore\b", r"\bterminal\b",
    r"\bevening\b", r"\bmorning\b", r"\bnight\b", r"\bweekday\b", r"\bweekend\b",
    r"\bdinner time\b", r"\blunch time\b", r"\bbreakfast\b",
    r"\bbank transfer\b", r"\bcontactless\b", r"\bcash\b", r"\bach\b"
]
FILLER_RE = re.compile("|".join(FILLER_WORDS), re.I)

# Common banking aliases to merge near-duplicates
BANK_ALIASES = {
    "chase bank": "chase",
    "wells fargo bank": "wells fargo",
    "bank of america": "bank of america",
    "citibank": "citi",
    "capital one bank": "capital one"
}

def core_merchant(text: str) -> str:
    """
    Derives normalized merchant-group key for leakage-safe splitting.
    Strips brackets, TXN numbers, store IDs, country suffixes, digits, and filler words.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    s = text.lower()
    # 1. Strip brackets and content (e.g. (Cash), (Contactless))
    s = BRACKETS_RE.sub(" ", s)
    # 2. Strip country suffixes (e.g. - USA, - CANADA)
    s = COUNTRY_SUFFIX_RE.sub(" ", s)
    # 3. Strip TXN / reference / store numbers
    s = TXN_REF_RE.sub(" ", s)
    s = STORE_NUM_RE.sub(" ", s)
    # 4. Strip filler words
    s = FILLER_RE.sub(" ", s)
    # 5. Remove remaining digits and punctuation except apostrophes/hyphens
    s = re.sub(r"[0-9#\-\/\\]+", " ", s)
    s = re.sub(r"[^\w\s']", " ", s)
    # 6. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 7. Check aliases
    if s in BANK_ALIASES:
        s = BANK_ALIASES[s]

    return s if s else "unknown"

def prepare_100k_sample() -> pd.DataFrame:
    """
    Samples exactly 100k records from mitulshah/transaction-categorization using shuffle(seed=1).
    Keeps only transaction_description and category.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_100K_PATH.exists():
        print(f"Loading existing 100k sample from {SAMPLE_100K_PATH}...")
        df = pd.read_csv(SAMPLE_100K_PATH)
        return df

    print(f"Creating 100k sample from {RAW_PARQUET} using Hugging Face datasets shuffle(seed=1)...")
    ds = Dataset.from_parquet(str(RAW_PARQUET))
    sampled_ds = ds.shuffle(seed=1).select(range(100000))
    df = sampled_ds.to_pandas()[["transaction_description", "category"]]
    df.to_csv(SAMPLE_100K_PATH, index=False)
    print(f"Saved 100k sample to {SAMPLE_100K_PATH} (Shape: {df.shape})")
    return df

def run_pre_training_checks(df: pd.DataFrame) -> None:
    """Runs required pre-training diagnostic checks."""
    print("=" * 60)
    print("PRE-TRAINING DIAGNOSTIC CHECKS (Section 2.2)")
    print("=" * 60)

    # 1. Indian merchants check
    indian_matches = df["transaction_description"].str.contains(
        r"swiggy|zomato|jio|paytm|flipkart", case=False
    ).sum()
    print(f"1. Indian merchants present in dataset? {indian_matches} rows")

    # 2. Top 30 core values
    top30 = df["core"].value_counts().head(30)
    print("\n2. Top 30 core values:\n", top30)

    # 3. Group size statistics
    gs = df["core"].value_counts()
    top10_pct = round(gs.head(10).sum() / len(df) * 100, 2)
    print(f"\n3. Group size distribution: Median = {gs.median()}, Max = {gs.max()}, Top-10 groups hold {top10_pct}% of rows")

    # 4. Category distribution
    print("\nCategory distribution in 100k sample:\n", df["category"].value_counts())

def create_merchant_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits data using StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    grouped on core merchant.
    Fold 0 = val (10%), Folds 1-2 = test (20%), Folds 3-9 = train (70%).
    """
    if "core" not in df.columns:
        df["core"] = df["transaction_description"].apply(core_merchant)

    run_pre_training_checks(df)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    folds = np.zeros(len(df), dtype=int)

    # Generate folds
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["category"], groups=df["core"])):
        folds[val_idx] = fold_idx

    # Assign split names
    splits = np.full(len(df), "train", dtype=object)
    splits[folds == 0] = "val"
    splits[(folds == 1) | (folds == 2)] = "test"
    df["split"] = splits

    # Programmatic Verification & Leakage Assertions
    train_cores = set(df[df["split"] == "train"]["core"])
    val_cores = set(df[df["split"] == "val"]["core"])
    test_cores = set(df[df["split"] == "test"]["core"])

    assert len(train_cores.intersection(val_cores)) == 0, "LEAKAGE DETECTED: train and val share core merchants!"
    assert len(train_cores.intersection(test_cores)) == 0, "LEAKAGE DETECTED: train and test share core merchants!"
    assert len(val_cores.intersection(test_cores)) == 0, "LEAKAGE DETECTED: val and test share core merchants!"
    print("\n" + "=" * 60)
    print("ASSERTION PASSED: Zero core merchant overlap between train, val, and test splits!")
    print("=" * 60)

    # Print split breakdown
    split_counts = df["split"].value_counts()
    split_pcts = df["split"].value_counts(normalize=True) * 100
    for split_name in ["train", "test", "val"]:
        n_rows = split_counts[split_name]
        pct = split_pcts[split_name]
        n_merchants = df[df["split"] == split_name]["core"].nunique()
        print(f"Split '{split_name}': {n_rows:,} rows ({pct:.2f}%), {n_merchants:,} unique core merchants")

    # Check class coverage across all splits
    print("\nClass distribution per split:")
    print(pd.crosstab(df["category"], df["split"]))

    # Save to data/mitulshah_split.csv
    df.to_csv(SPLIT_CSV_PATH, index=False)
    print(f"\nSuccessfully saved split dataset to {SPLIT_CSV_PATH}")
    return df

if __name__ == "__main__":
    df_100k = prepare_100k_sample()
    create_merchant_split(df_100k)
