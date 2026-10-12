import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier

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

print("Checking LightGBM parameters & rounds:")
COMMON = dict(learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, n_jobs=-1)
lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=63, subsample_freq=1,
                          random_state=42, verbose=-1, **COMMON)
lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=True)])
print(f"LGBM best_iteration_: {lgbm.best_iteration_}")
print(f"LGBM Clean Test Acc: {accuracy_score(yte_clean, lgbm.predict(Xte_clean)):.4f}")

# Now check why XGBoost stopped or what its objective and label encoding were:
# Wait! In XGBClassifier, when early_stopping_rounds=30 is passed in constructor vs fit:
# In XGBoost:
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    tree_method="hist",
    early_stopping_rounds=30,
    random_state=42,
    objective="multi:softprob",
    eval_metric="mlogloss",
    **COMMON
)
xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=30)
print(f"XGB best_iteration: {xgb.best_iteration}")
print(f"XGB Clean Test Acc: {accuracy_score(yte_clean, xgb.predict(Xte_clean)):.4f}")
