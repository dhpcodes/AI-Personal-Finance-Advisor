"""
Evaluate XGBoost with early stopping and same tree budget as LightGBM.
Tests whether 72.6% changes with early stopping on validation split.
"""

import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import TextPreprocessor

def main():
    print("Loading data...")
    df_split = pd.read_csv("data/mitulshah_split.csv")
    train_base = df_split[df_split.split == "train"].copy()
    val_df = df_split[df_split.split == "val"].copy()
    test_df = df_split[df_split.split == "test"].copy()
    df_seed = pd.read_csv("data/indian_merchants_train_seed.csv")

    train_df = pd.concat([train_base[["transaction_description", "category"]], df_seed[["transaction_description", "category"]]], ignore_index=True)
    preprocessor = TextPreprocessor(use_canonicalization=True)
    train_clean = [preprocessor.clean_text(t) for t in train_df.transaction_description]
    val_clean = [preprocessor.clean_text(t) for t in val_df.transaction_description]
    test_clean = [preprocessor.clean_text(t) for t in test_df.transaction_description]

    le = LabelEncoder().fit(train_df.category)
    y_tr = le.transform(train_df.category)
    y_va = le.transform(val_df.category)
    y_te = le.transform(test_df.category)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=3000, sublinear_tf=True, lowercase=True)
    X_tr = vectorizer.fit_transform(train_clean)
    X_va = vectorizer.transform(val_clean)
    X_te = vectorizer.transform(test_clean)

    print("Fitting XGBoost (n_estimators=100, learning_rate=0.15, max_depth=6, early_stopping_rounds=10)...")
    t0 = time.time()
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=10,
        n_estimators=100,
        learning_rate=0.15,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=10
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    fit_time = time.time() - t0
    val_preds = xgb_model.predict(X_va)
    test_preds = xgb_model.predict(X_te)

    val_acc = accuracy_score(y_va, val_preds)
    val_f1 = f1_score(y_va, val_preds, average="macro")
    test_acc = accuracy_score(y_te, test_preds)
    test_f1 = f1_score(y_te, test_preds, average="macro")

    print(f"Fit time: {fit_time:.2f}s | Best iteration: {xgb_model.best_iteration}")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%) | Val Macro-F1: {val_f1:.4f}")
    print(f"Clean Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%) | Test Macro-F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
