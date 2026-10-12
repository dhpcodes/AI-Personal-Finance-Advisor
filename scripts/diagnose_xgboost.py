import time
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import lightgbm as lgb

SPLIT_PATH = Path("data/mitulshah_split.csv")
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

print("Loading data...")
df = pd.read_csv(SPLIT_PATH)
train = df[df.split == "train"].copy()
val   = df[df.split == "val"].copy()
test  = df[df.split == "test"].copy()

r_tr = random.Random(42)
m_tr = np.random.RandomState(42).rand(len(train)) < 0.35
train.loc[m_tr, "transaction_description"] = [add_noise(t, r_tr, TRAIN_KINDS) for t in train.loc[m_tr, "transaction_description"]]

val_noisy = val.copy()
r_va = random.Random(11)
m_va = np.random.RandomState(1).rand(len(val)) < 0.35
val_noisy.loc[m_va, "transaction_description"] = [add_noise(t, r_va, TRAIN_KINDS) for t in val_noisy.loc[m_va, "transaction_description"]]

test_clean = test.copy()

le = LabelEncoder().fit(df.category)
print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {le.classes_}")

tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=3,
    max_features=20000,
    sublinear_tf=True,
    lowercase=False
)
Xtr = tfidf.fit_transform(train.transaction_description).astype(np.float32)
ytr = le.transform(train.category)
Xva = tfidf.transform(val_noisy.transaction_description).astype(np.float32)
yva = le.transform(val_noisy.category)
Xte_clean = tfidf.transform(test_clean.transaction_description).astype(np.float32)
yte_clean = le.transform(test_clean.category)

print(f"Xtr shape: {Xtr.shape}, non-zeros: {Xtr.nnz}")

# Check original XGBoost parameters
COMMON = dict(learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, n_jobs=-1)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    tree_method="hist",
    early_stopping_rounds=30,
    random_state=42,
    **COMMON
)

print("\n--- Diagnostic 1: Original XGBoost Training ---")
t0 = time.time()
xgb.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=25)
print(f"Fit time: {time.time() - t0:.2f}s")
print(f"Best iteration: {xgb.best_iteration}")
print(f"Best score: {xgb.best_score}")

train_pred = xgb.predict(Xtr)
val_pred = xgb.predict(Xva)
test_pred = xgb.predict(Xte_clean)

print(f"Train Accuracy: {accuracy_score(ytr, train_pred):.4f}")
print(f"Val Accuracy: {accuracy_score(yva, val_pred):.4f}")
print(f"Test Clean Accuracy: {accuracy_score(yte_clean, test_pred):.4f}")

# Diagnostic 2: What happens with objective='multi:softprob', num_class=10, eval_metric='mlogloss', without colsample_bytree=0.5 or higher n_estimators?
print("\n--- Diagnostic 2: Checking objective, eval_metric, colsample ---")
for colsample in [0.5, 0.8, 1.0]:
    for lr in [0.1, 0.2]:
        xgb_test = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=colsample,
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
        xgb_test.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p_val = xgb_test.predict(Xva)
        p_te = xgb_test.predict(Xte_clean)
        print(f"colsample={colsample}, lr={lr} (100 rounds) -> Val Acc: {accuracy_score(yva, p_val):.4f}, Test Clean Acc: {accuracy_score(yte_clean, p_te):.4f}")
