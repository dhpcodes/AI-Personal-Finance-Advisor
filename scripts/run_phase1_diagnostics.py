import sys
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

SPLIT_PATH = Path("data/mitulshah_split.csv")
RAW_SAMPLE_PATH = Path("data/mitulshah_sample_100k.csv")
TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]

def add_noise(text: str, rng: random.Random, kinds: list) -> str:
    ref = rng.randint(10**8, 10**12)
    vpa = re.sub(r"[^a-z]", "", text.lower())[:10] or "merchant"
    kind = rng.choice(kinds)
    if kind == "upi":   return f"UPI/{ref}/{text.upper()}/{vpa}@okaxis"
    if kind == "neft":  return f"NEFT-{ref}-{text.upper()}"
    if kind == "pos":   return f"POS {rng.randint(1000,9999)}XXXX{rng.randint(1000,9999)} {text.upper()} MUMBAI"
    if kind == "imps":  return f"IMPS/{ref}/{text.upper()}/REF{rng.randint(100,999)}"
    if kind == "upper": return text.upper()
    if kind == "lower": return text.lower()
    if kind == "trunc": return text.upper()[: max(5, len(text) // 2)] + "..."
    s = list(text)
    if len(s) > 4:
        i = rng.randint(1, len(s) - 2)
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)

print("--- 1b. INVESTIGATING LEAKY BASELINE (ROW-LEVEL RANDOM SPLIT) ---")
# Let's inspect the dataset and check why random row split gave 98.56% vs 98.77%
df = pd.read_csv(SPLIT_PATH)
print(f"Total rows in mitulshah_split.csv: {len(df)}")
print(f"Unique core merchants: {df['core'].nunique()}")
print(f"Unique transaction descriptions: {df['transaction_description'].nunique()}")

# If we do standard 80/20 random row split on df without merchant grouping:
train_leaky, test_leaky = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])
overlap_merchants = set(train_leaky["core"]).intersection(set(test_leaky["core"]))
overlap_txns = set(train_leaky["transaction_description"]).intersection(set(test_leaky["transaction_description"]))
print(f"Random split merchant overlap: {len(overlap_merchants)} / {test_leaky['core'].nunique()} test merchants ({len(overlap_merchants)/test_leaky['core'].nunique()*100:.1f}%)")
print(f"Random split exact text overlap: {len(overlap_txns)} / {test_leaky['transaction_description'].nunique()} unique test texts ({len(overlap_txns)/test_leaky['transaction_description'].nunique()*100:.1f}%)")

# Train LogReg on clean train_leaky, evaluate on test_leaky:
tfidf_leaky = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
Xtr_leak = tfidf_leaky.fit_transform(train_leaky["transaction_description"])
Xte_leak = tfidf_leaky.transform(test_leaky["transaction_description"])
lr_leak = LogisticRegression(C=10, max_iter=300, random_state=42)
lr_leak.fit(Xtr_leak, train_leaky["category"])
pred_leak = lr_leak.predict(Xte_leak)
acc_leak = accuracy_score(test_leaky["category"], pred_leak)
print(f"Leaky random split LogReg Clean Accuracy: {acc_leak:.5f}")

# Now compare with group split:
train_grp = df[df.split == "train"].copy()
test_grp = df[df.split == "test"].copy()
# On group split, what is overlap?
overlap_grp_m = set(train_grp["core"]).intersection(set(test_grp["core"]))
print(f"Merchant-group split merchant overlap: {len(overlap_grp_m)} (LEAKAGE FREE!)")

tfidf_grp = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=20000, sublinear_tf=True, lowercase=False)
Xtr_grp = tfidf_grp.fit_transform(train_grp["transaction_description"])
Xte_grp = tfidf_grp.transform(test_grp["transaction_description"])
lr_grp = LogisticRegression(C=10, max_iter=300, random_state=42)
lr_grp.fit(Xtr_grp, train_grp["category"])
pred_grp_clean = lr_grp.predict(Xte_grp)
acc_grp_clean = accuracy_score(test_grp["category"], pred_grp_clean)
print(f"Merchant-group split LogReg Clean Test Accuracy: {acc_grp_clean:.5f}")

# Generate noisy test set for group split:
r_te = random.Random(7)
test_grp_noisy = test_grp.copy()
TEST_KINDS = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]
test_grp_noisy["transaction_description"] = [add_noise(t, r_te, TEST_KINDS) for t in test_grp_noisy["transaction_description"]]
Xte_grp_noisy = tfidf_grp.transform(test_grp_noisy["transaction_description"])
pred_grp_noisy = lr_grp.predict(Xte_grp_noisy)
acc_grp_noisy = accuracy_score(test_grp["category"], pred_grp_noisy)
print(f"Merchant-group split LogReg NOISY Test Accuracy: {acc_grp_noisy:.5f}")

print("\n--- 1a. INVESTIGATING XGBOOST ---")
# Let's inspect XGBoost training rounds, early stopping, and objective
val_grp = df[df.split == "val"].copy()
r_va = random.Random(11)
m_va = np.random.RandomState(1).rand(len(val_grp)) < 0.35
val_grp_noisy = val_grp.copy()
val_grp_noisy.loc[m_va, "transaction_description"] = [add_noise(t, r_va, TRAIN_KINDS) for t in val_grp_noisy.loc[m_va, "transaction_description"]]

le = LabelEncoder().fit(df.category)
ytr = le.transform(train_grp.category)
yva = le.transform(val_grp_noisy.category)
yte = le.transform(test_grp.category)
Xva_grp = tfidf_grp.transform(val_grp_noisy.transaction_description)

# Check XGBoost with small sample to see convergence speed and learning rate
COMMON = dict(learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, n_jobs=-1)
# Note: In XGBoost, tree_method='hist' with 20000 sparse features and colsample_bytree=0.5:
# Why did it get 54.6% in compare_v2_seeds.csv?
# Let's inspect the original fitted XGBoost or check how early stopping behaved:
xgb_diag = XGBClassifier(
    n_estimators=60, # Original ran 60 rounds in ~190s
    max_depth=6,
    tree_method="hist",
    early_stopping_rounds=30,
    random_state=42,
    **COMMON
)
xgb_diag.fit(Xtr_grp, ytr, eval_set=[(Xtr_grp, ytr), (Xva_grp, yva)], verbose=10)
print(f"xgb_diag best_iteration: {xgb_diag.best_iteration}")
tr_acc = accuracy_score(ytr, xgb_diag.predict(Xtr_grp))
va_acc = accuracy_score(yva, xgb_diag.predict(Xva_grp))
te_acc = accuracy_score(yte, xgb_diag.predict(Xte_grp))
print(f"xgb_diag (60 rounds) -> Train Acc: {tr_acc:.4f}, Val Acc: {va_acc:.4f}, Test Clean Acc: {te_acc:.4f}")
