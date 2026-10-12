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

## 3. Verified Benchmark Results

### 3.1 7-Model Comparison Matrix

| Model ID | Architecture | Status | Clean Test Acc | Clean Macro-F1 | Noisy Test Acc | Noisy Macro-F1 | Train Time (s) | Resource Requirements |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | **TF-IDF + Logistic Regression** | **MEASURED** | 0.9877 | 0.9877 | **0.8116** | **0.8104** | 10.5 | CPU (Lightweight) |
| **Model 2** | **TF-IDF + Calibrated Linear SVM** | **MEASURED** | **0.9893** | **0.9893** | 0.7823 | 0.7811 | 15.6 | CPU (Lightweight) |
| **Model 3** | TF-IDF + LightGBM (Matched Baseline) | **MEASURED** | 0.9568 | 0.9565 | 0.7158 | 0.7218 | 458.6 | CPU Multicore |
| **Model 3b** | Stacked LightGBM (Tuned + OOF Meta-Features) | **MEASURED** | 0.9822 | 0.9821 | 0.7579 | 0.7588 | 133.7 | CPU Multicore |
| **Model 4** | TF-IDF + XGBoost (Matched Baseline) | **MEASURED** | 0.5462 | 0.5427 | 0.4074 | 0.4252 | 193.4 | CPU Multicore |
| **Model 4b** | Stacked XGBoost (Tuned + OOF Meta-Features) | **MEASURED** | 0.9850 | 0.9849 | 0.7609 | 0.7624 | 536.9 | CPU Multicore |
| **Model 5** | Sentence Embeddings (`all-MiniLM-L6-v2`) | **NOT RUN** | - | - | - | - | - | Requires T4 GPU (`notebooks/04b_sentence_embeddings.py`) |
| **Model 6** | Fine-tuned ModernBERT / DistilBERT | **NOT RUN** | - | - | - | - | - | Requires T4/A100 GPU (`notebooks/05_modernbert_training.py`) |
| **Model 7** | Llama 3.2 1B Instruct QLoRA | **NOT RUN** | - | - | - | - | - | Requires A100 GPU (`notebooks/06_llama_qlora_training.py`) |

*Finding on Tree Models vs Linear Models*: Linear models directly accumulate evidence across sparse n-gram features and generalize far better under corruption (81.16% vs 76.09% noisy test acc) while training in 10–15 seconds compared to minutes for tree ensembles.

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

## 4. Statistical Honesty & Caveats

1. **Small Real Indian Test Set (`own_test_indian.csv`, $N=105$)**:
   - Model-Only: **50.48%** (95% Wilson Score CI: `[41.07%, 59.85%]`)
   - Model + Keyword Rule: **55.24%** (95% Wilson Score CI: `[45.71%, 64.40%]`)
   - Difference (+4.76%, 5 transactions) is **within the confidence interval** ($p = 0.4894$). We cannot statistically claim that keyword rules improve accuracy over the ML model on this test set. Expanding this sample to $N \ge 1,000$ real transactions is future work.
2. **Synthetic Noise Family Sharing**: Both noise generation and canonicalization target Indian banking formats (UPI, NEFT, IMPS, POS). While IMPS and NEFT were strictly held out during training, within-family testing has inherent structural overlap.
3. **Forecast Seasonality**: Synthetic spending series contain a hardcoded October Diwali multiplier that Prophet naturally fits with high fidelity ($R^2 > 0.90$). On real data, festive shifts follow the lunisolar calendar and exhibit greater discretionary variance.

---

## 5. Getting Started & Reproducibility

### 5.1 Installation
Clone the repository and install requirements in Python 3.10+:
```bash
git clone https://github.com/dhpcodes/AI-Personal-Finance-Advisor.git
cd AI-Personal-Finance-Advisor
pip install -r requirements.txt
```

### 5.2 Running the Full Test Suite
Execute all 74 unit, integration, privacy, and regression tests:
```bash
pytest -o pythonpath=.
```
Expected output:
```
74 passed in ~15s
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

## 6. Repository Structure

```
├── app/
│   └── app.py                     # Interactive Streamlit application
├── data/
│   ├── mitulshah_split.csv        # Deduplicated unseen merchant split
│   └── own_test_indian.csv        # Held-out real Indian bank statements (N=105)
├── models/
│   └── final_model.joblib         # Calibrated Linear SVM production artifact
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
└── tests/                         # Comprehensive pytest test suite (74 tests)
```

---

## 7. License
MIT License. See [LICENSE](LICENSE) for details.
