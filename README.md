# Noise-Robust, Leakage-Safe Transaction Categorization for Indian Banking

An evidence-based, privacy-first, and grounded **AI Personal Finance Advisor** built with Python, Scikit-Learn, LightGBM, XGBoost, pandas, and Streamlit.

> **Core Thesis**:
> *"Noise-robust, leakage-safe, personalizable transaction categorization for Indian bank statements, with a grounded advisor."*
> The machine learning benchmarks serve as supporting evidence for this thesis, not the headline.

---

## 1. System Architecture

```
                       RAW INDIAN BANK STATEMENT
                                   │
                                   ▼
                  DOMAIN CANONICALIZER (src/canonicalize.py)
   (Strips UPI VPAs, POS slips, NEFT prefixes, IMPS IDs, masked cards)
                                   │
                                   ▼
                   TIERED PERSONALIZED CLASSIFIER
   ┌───────────────────────────────────────────────────────────────┐
   │ Level 1: Global Calibrated Linear SVM (Fallback baseline)     │
   │ Level 2: User Correction Memory (Exact / Normalized match)    │
   │ Level 3: Adaptive Cosine k-NN Layer (Active at N >= 5 edits)   │
   └───────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                      VERIFIED TRANSACTION LABELS
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
         BEHAVIORAL ENGINE              DETERMINISTIC ANALYTICS
    (Cash Flow & Discretionary)    (Budgeting, Goals, Forecasting)
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
               OUTBOUND PRIVACY FIREWALL (src/privacy.py)
      (38-Field Allowlist + Indian PII Redaction: PAN, Aadhaar, Phone)
                                   │
                                   ▼
               DETERMINISTIC PYTHON FINANCIAL TOOLS
         (Spending aggregations, anomaly detection, forecasts)
                                   │
                                   ▼
            NUMERIC GROUNDING VALIDATOR (src/numeric_validator.py)
    (Asserts every number, %, and currency matches tool-computed facts)
                                   │
                                   ▼
          GROUNDED FINANCIAL ADVISOR (src/grounded_llm.py)
      (Provider-agnostic: Ollama / OpenAI-compatible / Offline Fallback)
```

---

## 2. Key Architectural Contributions

### 1. Adaptive NLP & Domain Canonicalization (`src/canonicalize.py`, `src/adaptive_nlp.py`)
- **Domain Canonicalization**: Standardizes Indian banking transaction noise patterns (`UPI/123/SWIGGY/vpa@okaxis` $\rightarrow$ `SWIGGY`, `POS 4129XXXX9812 RELIANCE MUMBAI` $\rightarrow$ `RELIANCE`, `NEFT-928-RENT` $\rightarrow$ `RENT`). Boosts accuracy by **+22.20%** on held-out IMPS and **+12.98%** on held-out NEFT without model retraining.
- **Multi-Tier Personalization**: Combines the global classifier with user-specific correction memory and prototype k-NN. Increases user accuracy from **38.85%** ($N=0$) to **88.89%** ($N=20$) while maintaining strict multi-tenant isolation.

### 2. Behavioral Modeling & Discretionary Tracking (`src/behavioral_engine.py`)
- Computes individual cash flow ratios, recurring transaction cadence, and discretionary spending distributions (Dining, Entertainment, Shopping).

### 3. Personalized Forecasting & Backtesting (`src/forecast.py`)
- 1-step-ahead spending forecasting with 80% empirical uncertainty intervals.
- Validated via rolling-origin backtesting across multi-month financial profiles.

### 4. Explainable AI & Statistical Calibration (`src/explainability.py`, `src/calibration.py`)
- N-gram feature weights for every prediction.
- Post-hoc temperature scaling ($T=0.6453$) fitted on noisy validation data, reducing Expected Calibration Error (ECE) from 0.1459 to 0.1321.
- Confidence-based risk-coverage routing to flag uncertain transactions for human review.

### 5. Grounded Advisor & Numeric Hallucination Defense (`src/numeric_validator.py`, `src/grounded_llm.py`)
- **Strict Grounding**: Deterministic Python functions compute financial numbers; the LLM merely translates verified data into natural language.
- **Numeric Grounding Validator**: Automatically extracts every scalar, currency amount, and percentage in the LLM response and ensures it exists in tool data. Caught **100% of injected numerical hallucinations** (25/25) during evaluation, falling back cleanly to deterministic templates.
- **Privacy Firewall (`PrivacyGuard`)**: Restricts outbound payloads to a 38-field allowlist and automatically redacts Indian PAN, Aadhaar, credit card, and mobile phone numbers.

---

## 3. Verified Multi-Model Benchmark Results

Evaluated on identical zero-leakage partitions ($N=20,040$ held-out test split, $N=105$ real Indian bank statements) from the Single Source of Truth ([`results/final_metrics.csv`](results/final_metrics.csv)):

### 3.1 4-Model Family Comparison Matrix

| Model Architecture | Model Family | Status | Clean Test Acc | 95% Wilson CI | Noisy V2 Acc (Headline) | Typo Acc | Indian Model-Only (N=105 / 96 / 58) | Latency (ms) | Fit Time | Artifact Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production Baseline** (`final_model.joblib`) | Linear (Legacy) | **MEASURED** | 0.9907 | `[98.92%, 99.19%]` | 0.6389 *(init: 0.9273)* | 0.6810 | 0.5619 / — / — | 4.76 ms | *Legacy* | 10.69 MB |
| **Candidate Linear SVM** (`variant_b_candidate.joblib`) | Linear (SVM) | **MEASURED** | **0.9900** | `[98.85%, 99.13%]` | **0.7639** *(init: 0.9719)* | **0.8953** | **0.8762 / 0.8646 / 0.8103** | 4.96 ms | 46.6s | 8.29 MB |
| **LightGBM Classifier** (`lightgbm_...joblib`) | Boosted Trees (GBDT) | **MEASURED** | **0.9772** | `[97.50%, 97.92%]` | **0.6355** *(init: 0.9309)* | **0.8054** | **0.7810 / 0.8021 / 0.7414** | **1.97 ms** | 132.9s | 6.96 MB |
| **XGBoost Classifier** (`xgboost_...joblib`) | Boosted Trees (GBDT) | **MEASURED** | 0.7261 | `[71.99%, 73.23%]` | 0.4514 *(init: 0.6689)* | 0.5373 | 0.7048 / — / — | **1.55 ms** | 265.1s | 1.47 MB |
| **MiniLM-L6-v2 + LogReg** (`minilm_...joblib`) | Dense Embeddings | **MEASURED** | 0.8967 | `[89.24%, 90.08%]` | 0.4781 *(init: 0.8282)* | 0.5706 | 0.8095 / — / — | 9.73 ms | 12.0s | 0.02 MB |
| **DistilBERT Fine-Tuned** (`notebooks/05_...`) | Transformer | **MEASURED** | **0.9913** | `[98.99%, 99.25%]` | **0.6900** | 0.7752 | **0.9333 / 0.9271 / 0.8793** | (GPU batch) | 348.5s | 255.4 MB |

*Notes on evaluation columns*:
- **Noisy Test**: Headline noisy accuracy is evaluated on **Noisy V2** (independent noise generator, $N=20,040$). Initial noisy test scores *(in parentheses)* shared noise generator logic with training and are described only as initial, optimistic.
- **Indian Statement Acc**: Primary reported metric is **Model-Only** accuracy across $N=105$ (full real test), $N=96$ (non-overlapping), and $N=58$ (fully disjoint). Rule-assisted (+Rent) accuracies are secondary: Linear SVM (+Rent): 92.38% ($N=105$), 91.67% ($N=96$), 89.66% ($N=58$); DistilBERT (+Rent): 98.10% ($N=105$), 97.92% ($N=96$), 96.55% ($N=58$); LightGBM (+Rent): 82.86% ($N=105$), 85.42% ($N=96$), 82.76% ($N=58$). *The rent rule is an offline post-classification heuristic used in evaluation scripts only. It is not applied in the deployed Streamlit app, whose rent transactions may be misclassified (e.g. 'MONTHLY APARTMENT RENT TRANSFER TO LANDLORD' → Government & Legal). Its keywords match the Rent rows of the test set, so +rent figures are not evidence of generalization. Model-only figures are the headline numbers.*
- **N=96 McNemar exact binomial tests** (from `results/n96_stats.csv`; 5 comparisons, not multiple-comparison corrected; treat as suggestive):
  | Comparison | b (n₁₀) | c (n₀₁) | exact p | Bonferroni (α=0.01) |
  |---|---|---|---|---|
  | DistilBERT vs Candidate SVM | 6 | 0 | **0.03125** | not significant |
  | DistilBERT vs LightGBM | 15 | 0 | **0.00006** | significant |
  | DistilBERT vs MiniLM | 13 | 0 | **0.00024** | significant |
  | SVM vs LightGBM | 11 | 2 | 0.02246 | not significant |
  | SVM vs MiniLM | 11 | 4 | 0.11847 | not significant |
  After Bonferroni correction (α=0.01 for 5 comparisons): only DistilBERT vs LightGBM and DistilBERT vs MiniLM remain significant; DistilBERT vs Candidate SVM does not.
- **Wilson 95% CIs on N=96 model-only** (exact counts: 89/96, 83/96, 74/96, 76/96): DistilBERT **[85.71%, 96.42%]**, Candidate SVM **[78.20%, 91.91%]**, LightGBM (20k artifact) **[67.73%, 84.35%]**, MiniLM **[70.00%, 86.09%]**.

*Architectural Findings by Data Type*:
- **Linear Models (SVM)**: High-dimensional character n-grams are naturally linearly separable. The Candidate Linear SVM achieves the highest robustness on independent character noise (76.39% Noisy V2), outperforming DistilBERT (69.00%), with strong typo robustness (89.53%) and clean accuracy (99.00%). DistilBERT leads on clean text (99.13%, p=0.00084 vs SVM).
- **Boosted Trees (LightGBM vs XGBoost)**: LightGBM is the top-performing boosted tree model (+25.11% clean accuracy over XGBoost; 1.97 ms vs 4.96 ms Linear SVM latency). Controlled experiments reveal that XGBoost's deficit was mostly due to sparse zero handling (+10 points when passed dense zeros) and depth-wise growth (+9 points with `grow_policy='lossguide'`); the remaining gap to LightGBM is unexplained. Exclusive Feature Bundling (EFB) remains an untested hypothesis.
- **Dense Embeddings (MiniLM)**: Captures semantic nuance on real Indian bank statements (80.95% model-only; 85.71% +rent — offline evaluation only, not deployed), but drops on typo noise (57.06%) because character substitutions break subword token representations.
- **Fine-Tuned Transformers (DistilBERT)**: Contextual self-attention achieves peak clean test accuracy (99.13%) and strong Indian statement performance (93.33% model-only; 98.10% +rent — offline evaluation only, not deployed), but drops to 69.00% under independent character noise with an ECE of 0.24, showing subword tokenizer fragility on character deletions. Fine-tuned on Kaggle in 348.5 s (effective batch 64 via 2x T4 GPUs, no gradient accumulation, lr 5e-5, 3 epochs, single seed 42).

**Deployed model**: Candidate Linear SVM (`variant_b_candidate.joblib`), chosen for robustness on independently generated noise (76.39% Noisy V2 vs 69.00% for DistilBERT) and low single-call latency (4.96 ms). DistilBERT performs best on clean text and the Indian statement set (preliminary — trained on the original seed containing 8 overlapping merchants); deploying it after retraining on the leakage-aware seed is future work.

### 3.2 Canonicalization Gain Per Noise Type

| Noise Corruption | Raw Input Acc | Canonicalized Acc | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **IMPS (Held-out from training)** | 67.24% | **89.44%** | **+22.20%** |
| **NEFT (Held-out from training)** | 76.59% | **89.57%** | **+12.98%** |
| **UPI** | 79.52% | **89.41%** | **+9.89%** |
| **POS** | 80.52% | **89.31%** | **+8.79%** |
| **UPPERCASE** | 80.89% | **88.94%** | **+8.05%** |
| **TRUNCATION** | 80.99% | **88.59%** | **+7.60%** |
| **TYPO / PERMUTATION** | 88.58% | **89.18%** | **+0.60%** |

---

## 4. Statistical Honesty & Integrity Audit

1. **Real Indian Test Set ($N=105$, `data/own_test_indian.csv`)**:
   - Primary Model-Only accuracies: DistilBERT: **93.33%** ($N=105$) / **92.71%** ($N=96$) / **87.93%** ($N=58$); Candidate Linear SVM: **87.62%** ($N=105$) / **86.46%** ($N=96$) / **81.03%** ($N=58$); MiniLM-LogReg: **80.95%** ($N=105$); LightGBM (3k-matched artifact `matched_lightgbm_3k.joblib`): **78.10%** ($N=105$) / **80.21%** ($N=96$) / **74.14%** ($N=58$); LightGBM (20k artifact `lightgbm_transaction_classifier.joblib`, used in app): **77.08%** ($N=96$); XGBoost: **70.48%** ($N=105$). Rule-assisted (+Rent) results are secondary.
   - McNemar statistical tests reveal that differences between top models on the $N=105$ sample have $p$-values between 0.12 and 0.65 ($p > 0.05$). On the N=96 non-overlapping subset, DistilBERT vs Candidate SVM: exact p=0.03125 (significant at α=0.05; not significant after Bonferroni correction; treat as suggestive).
2. **Synthetic Training Seed Provenance & Leakage Removal**:
   - `data/indian_merchants_train_seed.csv` (3,062 rows) was generated using templates of known Indian merchants.
   - 8 merchants overlapped between the training seed and the real test set; these were removed and retested for SVM, LightGBM, XGBoost, and MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.
3. **Synthetic Noise Function Sharing**:
   - The initial noisy test set shared noise code with training and is described only as "initial, optimistic" (e.g. Linear SVM 97.19%).
   - An independent noise generator was built for **Noisy V2** ($N=20,040$), which serves as the headline noisy benchmark (Linear SVM: 76.39%, DistilBERT: 69.00%, LightGBM: 63.55%).

---

## 5. Limitations

1. **Indian training seed provenance**: `data/indian_merchants_train_seed.csv` (3,062 rows) is LLM-generated from structured prompt templates; labels are unverified and not drawn from real bank statements.
2. **Small real Indian test set**: The real Indian test set is small ($N=105$); performance differences between top models on it are not statistically significant.
3. **Seed-test merchant overlap**: 8 merchants overlapped between the training seed and the real test set; these were removed and retested for SVM, LightGBM, XGBoost, and MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.
4. **Shared noise generator**: The first noisy test set shared noise code with training; an independent noisy test set (Noisy V2) was built to evaluate out-of-distribution robustness.
5. **Synthetic benchmark nature**: The Mitul Shah benchmark (100k transactions) is likely synthetic; clean accuracy near 99% reflects an easy, well-separated task.
6. **Single run variance**: Benchmarks use a single seed and a single run for all models; multi-seed variance was not evaluated across all architectures.
7. **Transformer calibration under noise**: DistilBERT is severely overconfident under character noise (Expected Calibration Error of 0.24 on Noisy V2).
8. **Rent rule origin and scope**: The rent rule is an offline post-classification heuristic used in evaluation scripts only. It is not applied in the deployed Streamlit app, whose rent transactions may be misclassified (e.g. 'MONTHLY APARTMENT RENT TRANSFER TO LANDLORD' → Government & Legal). Its keywords match the Rent rows of the test set, so +rent figures are not evidence of generalization. Model-only figures are the headline numbers.
9. **Ambiguous grocery labeling**: Quick-commerce grocery apps (BigBasket, Blinkit, Zepto) exhibit ambiguous ground truth labels across Food & Dining vs Shopping & Retail.
10. **Text-only inputs**: The classification engine is text-only and does not incorporate transaction amount, timestamp, or Merchant Category Codes (MCC).
11. **XGBoost gap not fully ablated**: Controlled experiments explain +10 pts (sparse zeros) and +9 pts (depth-wise growth), but the remaining gap to LightGBM is unexplained. Exclusive Feature Bundling (EFB) was cited as a hypothesis but not ablated.

---

## 6. Getting Started & Reproducibility

### 5.1 Installation
Clone the repository and install requirements in Python 3.10+:
```bash
git clone https://github.com/dhpcodes/AI-Personal-Finance-Advisor.git
cd AI-Personal-Finance-Advisor
pip install -r requirements.txt
```

### 5.2 Running the Full Test Suite
Execute all 106 unit, integration, privacy, and regression tests:
```bash
pytest -o pythonpath=.
```
Expected output:
```
106 passed in ~16s
```

### 5.3 Reproducing Benchmarks & Evaluations
```bash
# 1. Noise robustness ablation matrix & per-noise-type evaluation
python scripts/run_phase2_noise_robustness.py

# 2. Boosted tree tuning post-mortem (Optuna + Stacking)
python scripts/run_phase3_boosted_trees.py

# 3. Personalization simulation across N feedback steps
python scripts/run_phase5_personalization.py

# 4. Grounding validator & hallucination benchmark
python scripts/run_phase6_grounding_eval.py

# 5. Statistical honesty (Wilson CIs & multi-seed evaluation)
python scripts/run_phase7_seeds_and_stats.py
```

### 5.4 Running the Application
#### Option A: Pure Offline Mode (No LLM, No Internet)
The core financial advisor operates completely locally without an LLM or internet connection:
```bash
streamlit run app/app.py
```

#### Option B: Local LLM Mode via Ollama
Ensure [Ollama](https://ollama.ai) is running locally with Llama 3.2:
```bash
ollama run llama3.2:3b
```
Configure `.env`:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2:3b
```
Then launch the Streamlit app:
```bash
streamlit run app/app.py
```

---

## 7. Repository Structure

```
├── app/
│   └── app.py                     # Interactive Streamlit application
├── data/
│   ├── mitulshah_split.csv        # Deduplicated unseen merchant split
│   └── own_test_indian.csv        # Held-out real Indian bank statements (N=105)
├── models/
│   ├── final_model.joblib              # Production Baseline (legacy, read-only)
│   └── variant_b_candidate.joblib     # Candidate Linear SVM (20k features, app default)
├── notebooks/                     # Colab workflows for GPU models (ModernBERT, QLoRA)
├── results/                       # Measured metrics, ablation CSVs, and figures
│   ├── EXPERIMENT_REPORT.md       # Comprehensive experimental findings & audit
│   ├── model_comparison_consolidated.csv
│   ├── noise_ablation_matrix.csv
│   ├── per_noise_type_evaluation.csv
│   └── grounding_evaluation_report.csv
├── scripts/                       # Reproducible phase benchmark scripts
├── src/                           # Production modular architecture
│   ├── adaptive_nlp.py            # Multi-tier personalization layer
│   ├── behavioral_engine.py       # Cash flow & discretionary spending
│   ├── calibration.py             # Post-hoc temperature scaling & ECE
│   ├── canonicalize.py            # Domain-specific banking string normalization
│   ├── database.py                # Multi-tenant SQLite storage & user isolation
│   ├── forecast.py                # Time-series forecasting & backtesting
│   ├── grounded_llm.py            # Provider-agnostic grounded LLM interface
│   ├── numeric_validator.py       # Deterministic numerical hallucination validator
│   ├── predict.py                 # Core transaction classification engine
│   └── privacy.py                 # Outbound allowlist & PII redaction firewall
└── tests/                         # Comprehensive pytest test suite (106 tests)
```

---

## 8. License
MIT License. See [LICENSE](LICENSE) for details.
