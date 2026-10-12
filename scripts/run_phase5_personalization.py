"""Phase 5: Personalization Simulation Suite (Synthetic Users).

Evaluates adaptive NLP performance as a function of user correction count N:
- Generic Baseline (Level 1 Global Model)
- Memory-Only (Level 2 Exact / Token Match)
- Memory + Lightweight k-NN Model (Level 3 Multi-Tier Architecture)

Simulation Settings:
- K=5 synthetic user archetypes with consistent idiosyncratic preferences
- Streams drawn strictly from held-out validation/test pools (NO LEAKAGE from global train)
- Evaluated across corrections N in [0, 1, 3, 5, 10, 20]
- Evaluated over seeds [42, 43, 44], reporting mean ± std
"""

import sys
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.canonicalize import canonicalize

SPLIT_PATH = Path("data/mitulshah_split.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 5: PERSONALIZATION SIMULATION (SYNTHETIC USERS)")
    print("=" * 70)

    df = pd.read_csv(SPLIT_PATH)
    train = df[df.split == "train"].copy()
    val = df[df.split == "val"].copy()
    test = df[df.split == "test"].copy()

    # Fit global model strictly on training split
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        lowercase=False
    )
    X_tr = tfidf.fit_transform(train["transaction_description"])
    global_model = LogisticRegression(C=10, max_iter=300, random_state=42)
    global_model.fit(X_tr, train["category"])

    # Define K=5 synthetic user preference profiles (override rules)
    # These represent realistic personal classification divergences:
    USER_PREFERENCES = [
        {
            "user_id": "user_consultant",
            "rule_name": "Classifies Cafes & Dining as Business Utilities",
            "trigger_keywords": ["starbucks", "cafe", "coffee", "restaurant", "bistro"],
            "personalized_category": "Utilities & Services"
        },
        {
            "user_id": "user_contractor",
            "rule_name": "Classifies Retail Stores as Healthcare Supplies",
            "trigger_keywords": ["walmart", "target", "costco", "supermarket"],
            "personalized_category": "Healthcare & Medical"
        },
        {
            "user_id": "user_remote_dev",
            "rule_name": "Classifies Tech Services as Entertainment",
            "trigger_keywords": ["amazon", "aws", "cloud", "digital", "hosting"],
            "personalized_category": "Entertainment & Recreation"
        },
        {
            "user_id": "user_frequent_traveler",
            "rule_name": "Classifies Transit & Rides as Financial Services / Reimbursable",
            "trigger_keywords": ["uber", "lyft", "transit", "taxi", "airline"],
            "personalized_category": "Financial Services"
        },
        {
            "user_id": "user_philanthropist",
            "rule_name": "Classifies Community payments as Charity",
            "trigger_keywords": ["society", "community", "club", "association"],
            "personalized_category": "Charity & Donations"
        }
    ]

    # Combine validation and test sets as the personalization evaluation pool (NO train leakage)
    eval_pool = pd.concat([val, test]).reset_index(drop=True)
    N_CORRECTIONS_LIST = [0, 1, 3, 5, 10, 20]
    SEEDS = [42, 43, 44]

    sim_results = []

    for seed in SEEDS:
        rng = random.Random(seed)

        for u_idx, pref in enumerate(USER_PREFERENCES):
            u_id = pref["user_id"]
            target_cat = pref["personalized_category"]
            triggers = pref["trigger_keywords"]

            # Filter eval pool to transactions matching user's trigger merchants
            mask_trigger = eval_pool["transaction_description"].str.lower().apply(
                lambda s: any(k in s for k in triggers)
            )
            user_stream = eval_pool[mask_trigger].copy().sample(frac=1.0, random_state=seed).reset_index(drop=True)

            if len(user_stream) < 35:
                # If pool small, take all available
                pool_to_use = user_stream
            else:
                pool_to_use = user_stream.iloc[:50]

            # True label for this user is the personalized preference:
            y_user_true = [target_cat] * len(pool_to_use)
            eval_texts = pool_to_use["transaction_description"].tolist()

            # Global predictions (Level 1)
            X_eval = tfidf.transform(eval_texts)
            global_preds = global_model.predict(X_eval)

            for N in N_CORRECTIONS_LIST:
                if N == 0:
                    # Generic only
                    acc_generic = accuracy_score(y_user_true, global_preds)
                    acc_mem = acc_generic
                    acc_hybrid = acc_generic
                else:
                    # User provides N corrections on the first N items in their stream
                    feedback_texts = eval_texts[:N]
                    feedback_labels = [target_cat] * N

                    # Test on the remaining unseen items in user stream
                    test_eval_texts = eval_texts[N:]
                    y_test_true = y_user_true[N:]

                    if len(test_eval_texts) == 0:
                        continue

                    # 1. Generic Baseline
                    X_test_eval = tfidf.transform(test_eval_texts)
                    preds_generic = global_model.predict(X_test_eval)
                    acc_generic = accuracy_score(y_test_true, preds_generic)

                    # 2. Memory-Only (Exact / normalized string match against feedback memory)
                    canon_feedback = set([canonicalize(t).lower() for t in feedback_texts])
                    preds_mem = []
                    for t, g_pred in zip(test_eval_texts, preds_generic):
                        c_text = canonicalize(t).lower()
                        # Direct or substring match in memory
                        if any(c_text == mem or mem in c_text or c_text in mem for mem in canon_feedback):
                            preds_mem.append(target_cat)
                        else:
                            preds_mem.append(g_pred)
                    acc_mem = accuracy_score(y_test_true, preds_mem)

                    # 3. Hybrid: Memory + Lightweight k-NN Model (Level 3)
                    preds_hybrid = []
                    X_fb = tfidf.transform(feedback_texts)
                    knn = KNeighborsClassifier(n_neighbors=min(3, N), metric="cosine")
                    knn.fit(X_fb, feedback_labels)

                    # Transform test items
                    X_sub_test = tfidf.transform(test_eval_texts)
                    knn_preds = knn.predict(X_sub_test)
                    knn_distances, _ = knn.kneighbors(X_sub_test)
                    # Cosine distance to similarity: sim = 1 - dist
                    knn_sims = 1.0 - knn_distances.mean(axis=1)

                    for i, (t, g_pred) in enumerate(zip(test_eval_texts, preds_generic)):
                        c_text = canonicalize(t).lower()
                        # If exact/token memory matches, use memory
                        if any(c_text == mem or mem in c_text or c_text in mem for mem in canon_feedback):
                            preds_hybrid.append(target_cat)
                        elif N >= 3 and knn_sims[i] >= 0.50:
                            # Level 3 k-NN prototype generalization
                            preds_hybrid.append(knn_preds[i])
                        else:
                            # Fallback to Level 1
                            preds_hybrid.append(g_pred)

                    acc_hybrid = accuracy_score(y_test_true, preds_hybrid)

                sim_results.append({
                    "Seed": seed,
                    "User_Archetype": u_id,
                    "N_Corrections": N,
                    "Generic_Accuracy": acc_generic,
                    "Memory_Only_Accuracy": acc_mem,
                    "Hybrid_Adaptive_Accuracy": acc_hybrid
                })

    res_df = pd.DataFrame(sim_results)
    agg_df = res_df.groupby("N_Corrections").agg({
        "Generic_Accuracy": ["mean", "std"],
        "Memory_Only_Accuracy": ["mean", "std"],
        "Hybrid_Adaptive_Accuracy": ["mean", "std"]
    }).round(4)

    # Flatten columns
    agg_df.columns = ["Generic_Mean", "Generic_Std", "Memory_Mean", "Memory_Std", "Hybrid_Mean", "Hybrid_Std"]
    agg_df = agg_df.reset_index()
    agg_df["Data_Source"] = "SYNTHETIC USERS (Held-Out Val/Test Pools)"

    agg_df.to_csv(RESULTS_DIR / "adaptive_nlp_evaluation.csv", index=False)
    print("Saved results/adaptive_nlp_evaluation.csv")
    print(agg_df)

    # Generate Personalization Curve Plot
    plt.figure(figsize=(8.5, 5.5))
    n_pts = agg_df["N_Corrections"].values

    plt.plot(n_pts, agg_df["Generic_Mean"] * 100, "o--", color="#64748B", label="Generic Baseline (No Personalization)", linewidth=2)
    plt.plot(n_pts, agg_df["Memory_Mean"] * 100, "s-", color="#2563EB", label="Level 2: Memory-Only Match", linewidth=2)
    plt.plot(n_pts, agg_df["Hybrid_Mean"] * 100, "^-", color="#16A34A", label="Level 3: Multi-Tier (Memory + k-NN Prototype)", linewidth=2.5)

    plt.fill_between(n_pts, (agg_df["Hybrid_Mean"] - agg_df["Hybrid_Std"])*100, (agg_df["Hybrid_Mean"] + agg_df["Hybrid_Std"])*100, color="#16A34A", alpha=0.15)

    plt.title("Personalization Learning Curve: Accuracy vs Number of Corrections N", fontsize=12, fontweight="bold")
    plt.xlabel("Number of User Corrections Provided (N)", fontsize=11)
    plt.ylabel("Accuracy on User Stream (%)", fontsize=11)
    plt.xticks(n_pts)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "personalization_curve.png", dpi=200)
    plt.close()
    print("Saved plot to results/personalization_curve.png")
    print("\nPhase 5 Simulation Completed Successfully!")


if __name__ == "__main__":
    main()
