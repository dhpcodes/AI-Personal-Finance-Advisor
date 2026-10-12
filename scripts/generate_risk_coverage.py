import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

SPLIT_PATH = Path("data/mitulshah_split.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_KINDS = ["upi", "pos", "upper", "lower", "trunc", "typo"]
TEST_KINDS  = ["upi", "pos", "neft", "imps", "upper", "lower", "trunc", "typo"]

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

print("Loading data for Risk-Coverage Analysis...")
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
test_noisy = test.copy()
r_te = random.Random(7)
test_noisy["transaction_description"] = [add_noise(t, r_te, TEST_KINDS) for t in test_noisy["transaction_description"]]

le = LabelEncoder().fit(df.category)
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=3,
    max_features=20000,
    sublinear_tf=True,
    lowercase=False
)
Xtr = tfidf.fit_transform(train.transaction_description)
ytr = le.transform(train.category)

Xva = tfidf.transform(val_noisy.transaction_description)
yva = le.transform(val_noisy.category)

Xte_clean = tfidf.transform(test_clean.transaction_description)
yte_clean = le.transform(test_clean.category)

Xte_noisy = tfidf.transform(test_noisy.transaction_description)
yte_noisy = le.transform(test_noisy.category)

print("Fitting Logistic Regression (C=10, max_iter=300)...")
lr = LogisticRegression(C=10, max_iter=300, random_state=42)
lr.fit(Xtr, ytr)

val_proba = lr.predict_proba(Xva)
val_conf = val_proba.max(axis=1)
val_pred = val_proba.argmax(axis=1)

te_clean_proba = lr.predict_proba(Xte_clean)
te_clean_conf = te_clean_proba.max(axis=1)
te_clean_pred = te_clean_proba.argmax(axis=1)

te_noisy_proba = lr.predict_proba(Xte_noisy)
te_noisy_conf = te_noisy_proba.max(axis=1)
te_noisy_pred = te_noisy_proba.argmax(axis=1)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
rows = []
curve_val_cov, curve_val_acc = [], []
curve_clean_cov, curve_clean_acc = [], []
curve_noisy_cov, curve_noisy_acc = [], []

for tau in thresholds:
    # Selected on NOISY VALIDATION only:
    m_va = val_conf >= tau
    cov_va = float(m_va.mean())
    acc_va = float(accuracy_score(yva[m_va], val_pred[m_va])) if m_va.any() else 1.0

    # Evaluated at this tau on Clean Test and Noisy Test:
    m_tc = te_clean_conf >= tau
    cov_tc = float(m_tc.mean())
    acc_tc = float(accuracy_score(yte_clean[m_tc], te_clean_pred[m_tc])) if m_tc.any() else 1.0

    m_tn = te_noisy_conf >= tau
    cov_tn = float(m_tn.mean())
    acc_tn = float(accuracy_score(yte_noisy[m_tn], te_noisy_pred[m_tn])) if m_tn.any() else 1.0

    curve_val_cov.append(cov_va)
    curve_val_acc.append(acc_va)
    curve_clean_cov.append(cov_tc)
    curve_clean_acc.append(acc_tc)
    curve_noisy_cov.append(cov_tn)
    curve_noisy_acc.append(acc_tn)

    rows.append({
        "Confidence_Threshold_tau": tau,
        "Threshold_Selection_Dataset": "Noisy Validation (Tuning Only)",
        "Noisy_Val_Coverage_Pct": round(cov_va * 100, 2),
        "Noisy_Val_Accuracy": round(acc_va, 4),
        "Noisy_Val_Risk_Pct": round((1.0 - acc_va) * 100, 2),
        "Noisy_Test_Coverage_Pct": round(cov_tn * 100, 2),
        "Noisy_Test_Accuracy": round(acc_tn, 4),
        "Clean_Test_Coverage_Pct": round(cov_tc * 100, 2),
        "Clean_Test_Accuracy": round(acc_tc, 4),
        "Routing_Fallback_Pct_Noisy_Val": round((1.0 - cov_va) * 100, 2)
    })

res_df = pd.DataFrame(rows)
res_df.to_csv(RESULTS_DIR / "confidence_threshold_routing.csv", index=False)
print("Saved updated results/confidence_threshold_routing.csv")
print(res_df[["Confidence_Threshold_tau", "Noisy_Val_Coverage_Pct", "Noisy_Val_Accuracy", "Noisy_Test_Accuracy", "Clean_Test_Accuracy"]])

# Plot Risk-Coverage Curve
plt.figure(figsize=(9, 5.5))
plt.plot(np.array(curve_val_cov)*100, np.array(curve_val_acc)*100, "o-", label="Noisy Validation (Tuned)", color="#2563EB", linewidth=2)
plt.plot(np.array(curve_noisy_cov)*100, np.array(curve_noisy_acc)*100, "s--", label="Noisy Test (Held-Out NEFT/IMPS)", color="#DC2626", linewidth=2)
plt.plot(np.array(curve_clean_cov)*100, np.array(curve_clean_acc)*100, "^-.", label="Clean Test (Unseen Merchants)", color="#16A34A", linewidth=2)

plt.axvline(x=75.75, color="gray", linestyle=":", label="Selected Threshold (tau=0.80)")
plt.title("Risk-Coverage Curve: Accuracy vs Coverage Across Confidence Thresholds", fontsize=12, fontweight="bold")
plt.xlabel("Coverage (% of Transactions Auto-Classified)", fontsize=11)
plt.ylabel("Accuracy on Covered Transactions (%)", fontsize=11)
plt.ylim(75, 101)
plt.xlim(40, 102)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left", fontsize=10)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "risk_coverage_curve.png", dpi=200)
plt.close()
print("Saved plot to results/risk_coverage_curve.png")
