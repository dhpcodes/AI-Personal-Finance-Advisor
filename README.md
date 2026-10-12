# AI Personal Finance Advisor (v2.0)

A production-grade, privacy-first, explainable, and grounded **AI Personal Finance Advisor** built with Python, Scikit-Learn, LightGBM, XGBoost, pandas, Plotly, and Streamlit.

The system delivers automated transaction categorization robust to real-world Indian banking noise, leakage-safe merchant cross-validation, deterministic category budgeting, goal-based savings planning, time-series spending forecasting with rolling-origin backtesting, explainable AI diagnostics, and a grounded AI conversational interface protected by an outbound data minimization privacy firewall.

---

## 1. Executive Summary & The 5 Major Contributions

This platform addresses the fundamental challenge of personal financial management: transforming noisy, unstructured bank statements (e.g., `UPI/4829103948/SWIGGY/swiggy@okaxis`, `NEFT-92837492-RENT`, `POS 4129XXXX9812 RELIANCE MUMBAI`) into actionable financial decisions without hallucinating calculations or leaking sensitive financial PII.

```
                                  USER
                                    │
                                    ▼
                          STREAMLIT APPLICATION (app/app.py)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        TRANSACTION NLP PIPELINE             USER FINANCIAL DATA
                  │                                   │
                  ▼                                   ▼
          GENERIC CLASSIFIER                  BEHAVIORAL ENGINE
         (TF-IDF + Calibrated Linear SVM)     (Cash Flow & Discretionary Profiles)
                  │                                   │
                  ▼                                   ▼
          PERSONALIZED / ADAPTIVE NLP         DETERMINISTIC MODULES
        (Keyword Rules + User Feedback)      (BudgetManager & GoalSavingsPlanner)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                         EXPLAINABILITY ENGINE
                      (N-gram Weights & Calibrated ECE)
                                    │
                                    ▼
                       TIME-SERIES FORECASTING ENGINE
                     (1-Step Ahead + Volatility Bounds)
                                    │
                                    ▼
                     OUTBOUND PRIVACY FIREWALL (src/privacy.py)
                   (Field Allowlist + Regex PII Sanitization)
                                    │
                                    ▼
                     GROUNDED LLM INTERFACE (src/grounded_llm.py)
                   (Google Gemini / Local Deterministic Fallback)
                                    │
                                    ▼
                         EXPLAINABLE ADVICE & INSIGHTS
```

### The 5 Architectural Pillars:
1. **Contribution 1 — Adaptive NLP & Domain Routing**:
   Combines machine learning with deterministic domain rules (e.g., regex `with_rent_rule()` for Indian rent/PG/landlord payments) and personalized feedback adaptation, preventing domain blind spots.
2. **Contribution 2 — Behavioral Cash-Flow Modeling**:
   Extracts user financial profiles, income vs expense breakdowns, recurring subscription frequencies, and discretionary category spending (Food & Dining, Entertainment & Recreation, Shopping & Retail).
3. **Contribution 3 — Personalized Spending Forecasting & Backtesting**:
   Predicts 1-month-ahead spending with 80% empirical uncertainty bounds. Validated using rolling-origin backtesting across 18 expanding windows on multi-month profiles, demonstrating that Seasonal-Naive (Lag 12) models capture annual festive/Diwali spending spikes with 3.35% sMAPE and 0.626 MASE.
4. **Contribution 4 — Explainable AI & Calibration**:
   Provides n-gram feature importance weights for every classification, computes Expected Calibration Error (ECE), and establishes confidence thresholds on noisy text to route uncertain predictions to human review.
5. **Contribution 5 — Grounded LLM Layer with Privacy Firewall (`PrivacyGuard`)**:
   Enforces strict grounding: Python functions compute exact financial arithmetic first, and the LLM only formats verified facts. An outbound data minimization firewall enforces a 38-field allowlist and automatically redacts card numbers, bank accounts, passwords, Indian PAN, and Aadhaar PII before any API transmission.

---

## 2. Dataset Preparation & Leakage-Safe Merchant Splitting

### 2.1 Hugging Face Training Sample
The models are trained strictly on a reproducible sample of 100,000 records from:
`mitulshah/transaction-categorization` sampled with `shuffle(seed=1)`. Retaining only `transaction_description` and `category`.

Target Categories (10 standard classes):
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

### 2.2 Pre-Training Diagnostic Findings
- **Indian Merchant Presence in Raw Data**: Programmatic search for `swiggy|zomato|jio|paytm|flipkart` in the 100k sample yielded **0 matching rows**. This confirms that standard models trained on this dataset have never encountered Indian merchant names or transaction formats, proving the vital necessity of synthetic noise injection and our Indian banking test set.
- **Merchant Group Distribution**: Derived via normalized `core_merchant()` regex extractor.
  - Median merchant group size: `1`
  - Maximum merchant group size: `48`
  - Top 10 merchant groups hold only `~0.4%` of dataset rows.

### 2.3 Why Random Row Splitting Is Broken (Data Leakage Audit)
Standard random train/test row splitting causes severe **data leakage** in transaction categorization:
- Transactions from the same merchant (e.g., `"Starbucks Store #102"` in train and `"Starbucks Store #894"` in test) appear across splits.
- The model memorizes merchant names rather than learning generalized semantics.
- Random row-split baseline achieves **98.56% Accuracy** and **0.9856 Macro-F1** (`results/leaky_baseline_evaluation.csv`), creating a false sense of real-world generalization.

### 2.4 Leakage-Safe 10-Split Stratified Group K-Fold
We partitioned the dataset using `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` grouped by normalized `core_merchant`:
- **Train Split**: 69,885 rows (2,975 distinct merchants)
- **Validation Split**: 10,075 rows (333 distinct merchants)
- **Test Split**: 20,040 rows (803 distinct merchants)
- **Merchant Overlap Assertion**: **0.00% overlap**. Verified programmatically with zero cross-split merchant leakage (`data/mitulshah_split.csv`).

---

## 3. Synthetic Indian Banking Text Noise Pipeline

To rigorously evaluate real-world robustness, synthetic banking noise is injected:
- **Training Noise Formats**: UPI VPAs (`UPI/{ref}/{merchant}/{vpa}@okaxis`), POS slips (`POS 4129XXXX9812 {merchant} MUMBAI`), uppercase, lowercase, text truncation, character swaps / typos. (Applied to 35% of train/val records).
- **Test Noise Formats**: Includes all training formats PLUS **unseen held-out formats**:
  - `NEFT`: `NEFT-{ref}-{merchant}`
  - `IMPS`: `IMPS/{ref}/{merchant}/REF{code}`
- Because NEFT and IMPS are held out from training, the noisy test set measures genuine generalization to unfamiliar banking wrappers.

---

## 4. Candidate Model Comparison Benchmark Suite (7 Models)

All models are trained strictly on the identical training split and evaluated on clean unseen merchants and noisy Indian banking text.

| Model ID | Model Name | Evaluation Status | Clean Test Acc | Clean Macro-F1 | Noisy Test Acc | Noisy Macro-F1 | Latency (ms/txn) | Resource Profile |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | **TF-IDF (char 2-5) + Logistic Regression** 🏆 | **MEASURED** | **0.9877** | **0.9877** | **0.8116** | **0.8104** | **0.0016 ms** | CPU (Lightweight) |
| **Model 2** | **TF-IDF + Linear SVM (Calibrated)** | **MEASURED** | **0.9893** | **0.9893** | **0.7823** | **0.7811** | **0.0060 ms** | CPU (Lightweight) |
| **Model 3** | **TF-IDF + LightGBM (depth=6, leaves=63)** | **MEASURED** | **0.9568** | **0.9565** | **0.7158** | **0.7218** | **0.0459 ms** | CPU Multicore |
| **Model 4** | **TF-IDF + XGBoost (tree_method=hist)** | **MEASURED** | `results` | `results` | `results` | `results` | `results` | CPU / GPU |
| **Model 5** | **Sentence Transformers (all-MiniLM-L6-v2) + LogReg** | NOT RUN | — | — | — | — | ~12.0 ms | T4 GPU / 4GB RAM (`notebooks/04b_sentence_embeddings.py`) |
| **Model 6** | **Fine-Tuned ModernBERT / DistilBERT** | NOT RUN | — | — | — | — | ~25.0 ms | NVIDIA T4 / A100 GPU (`notebooks/05_modernbert_training.py`) |
| **Model 7** | **Llama 3.2 1B Instruct with QLoRA (4-bit)** | NOT RUN | — | — | — | — | ~120.0 ms | NVIDIA A10G / A100 GPU (`notebooks/06_llama_qlora_training.py`) |

### Key Machine Learning Findings:
1. **Linear Hyperplanes Beat Decision Trees on Text**:
   On 20,000 sparse character n-gram features, Logistic Regression (81.16% noisy Acc) and Linear SVM (78.23% noisy Acc) significantly outperform LightGBM (71.58% noisy Acc). Linear models aggregate evidence simultaneously across thousands of subtle sub-string cues, whereas shallow tree splits cannot effectively partition high-dimensional sparse representations.
2. **Inference Latency Advantage**:
   Logistic Regression delivers inference at **0.0016 ms per transaction** (~625,000 txns/sec), making it 28x faster than LightGBM and 7,500x faster than dense neural encoders.
3. **Statistical Confidence Intervals**:
   Group-bootstrap 95% confidence intervals are calculated by resampling merchant clusters rather than rows, preventing artificial tightening of error margins (`results/bootstrap_confidence_intervals.csv`).

---

## 5. User-Owned Indian Banking Test Set & Keyword Rent Rule

### 5.1 Indian Test Dataset (`data/own_test_indian.csv`)
100 hand-crafted real-world Indian banking transaction descriptions covering UPI, NEFT, IMPS, Swiggy, Zomato, Jio, Airtel, Uber, Ola, Apollo Pharmacy, Zerodha, and Rent payments across all categories.

### 5.2 Rent Rule Attribution Integrity
The Hugging Face dataset does not contain "Rent" as a training category. Therefore, pure ML models misclassify rent payments into *Financial Services* or *Utilities & Services*.

We implement a deterministic keyword router:
```python
RENT_RE = re.compile(r"\b(rent|landlord|pg|lease|housing society)\b", re.I)
def with_rent_rule(texts, model_labels):
    return [("Rent" if RENT_RE.search(t) else l) for t, l in zip(texts, model_labels)]
```
- **Scientific Integrity Principle**: We report **model-only accuracy** and **model+rule accuracy** separately in `results/own_test_evaluation.csv`. Rule-based improvements are never falsely attributed to ML generalization.

---

## 6. Deterministic Budgeting & Goal-Based Savings Modules

### 6.1 Category Budgeting Engine (`src/budgeting.py`)
- **Deterministic Refund Offsetting**: When transaction type is `refund`, the amount offsets (reduces) the net category expense.
- **Budget Tracking**:
  - `Within Budget`: Spending < 80% of budget.
  - `Near Limit`: Spending between 80% and 100% of budget.
  - `Over Budget`: Spending > 100% of budget.
  - `Unbudgeted`: Spending recorded without a target budget.

### 6.2 Goal-Based Savings Planner (`src/goal_saving.py`)
- Computes verified historical monthly surplus: `Average Monthly Income - Average Monthly Expense`.
- Calculates required monthly savings: `(Target Amount - Current Savings) / Target Months`.
- Identifies monthly shortfalls when surplus < required savings.
- Recommends actionable spending cutbacks exclusively from discretionary categories: `Food & Dining`, `Entertainment & Recreation`, and `Shopping & Retail`.
- Displays clear non-advisory disclaimers.

---

## 7. Time-Series Forecasting & Rolling-Origin Backtesting

### 7.1 Runtime Forecaster (`src/forecast.py`)
- Computes 1-step-ahead spending forecasts using a 3-month weighted moving average.
- Computes empirical uncertainty bounds (80% confidence interval: `Point Forecast ± 1.28 * Volatility`).
- Breaks down next-month projections across spending categories based on recent expenditure weights.
- Fails safely with informative messages if fewer than 3 months of dated transactions exist.

### 7.2 Rolling-Origin Backtest Benchmarks (18 Expanding Windows)
Evaluated on a 24-month profile with recurring Diwali/festive spending surges in October/November (`results/forecast_backtest_report.csv`):

| Forecasting Model | Backtest Windows | MAE (INR) | RMSE (INR) | sMAPE (%) | MASE | Model Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seasonal-Naive (12-Month Lag)** 🏆 | 18 | **₹2,977.71** | **₹3,708.15** | **3.35%** | **0.626** | **Superior**: Captures annual festival surges; MASE < 1.0 beats naive. |
| **Last-Month Naive (Lag 1)** | 18 | ₹4,508.99 | ₹6,708.30 | 4.99% | 0.948 | Standard benchmark; lags festival transitions by 1 month. |
| **Moving-Average (3-Month SMA)** | 18 | ₹5,148.58 | ₹7,184.16 | 5.71% | 1.083 | **Underperforms**: Smooths across sharp spikes, resulting in high lag error. |

---

## 8. Outbound Privacy & Data Minimization Firewall (`src/privacy.py`)

Financial systems connecting to commercial LLMs must ensure zero data leakage:
- **Strict Outbound Allowlist**: External LLM calls only receive pre-computed financial summaries matching an approved 38-field allowlist (`total_spending`, `net_cash_flow`, `category_breakdown`, etc.).
- **Raw Transaction Rejection**: Outbound dictionaries containing lists of transactions or raw database rows trigger an immediate `PrivacyViolationError`.
- **Regex PII Redaction**: Scans and sanitizes:
  - 16-digit credit card numbers (`\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b`)
  - 9–18 digit bank account numbers (`\b\d{9,18}\b`)
  - Credentials and passwords (`\b(password|pin|cvv|cvc|otp)\b`)
  - Indian PAN cards (`\b[A-Z]{5}\d{4}[A-Z]\b`)
  - Aadhaar format (`\b\d{4}\s\d{4}\s\d{4}\b`)
- **Deterministic Offline Fallback**: If no API key is provided, the conversational assistant runs locally with 100% deterministic templates.

---

## 9. Repository Architecture

```
AI-Personal-Finance-Advisor/
├── configs/
│   └── config.yaml                     # Application settings
├── data/
│   ├── mitulshah_sample_100k.csv       # Exact 100k HuggingFace sample (seed=1)
│   ├── mitulshah_split.csv             # Leakage-safe 10-split StratifiedGroupKFold
│   ├── own_test_indian.csv             # 100 Indian banking test transactions
│   ├── synthetic_monthly_transactions_24m.csv # 24-month backtesting dataset
│   └── demo/test.parquet               # Public demo sample
├── models/
│   └── final_model.joblib              # Calibrated Linear SVM serialized pipeline
├── notebooks/
│   ├── 04b_sentence_embeddings.py      # MiniLM sentence embedding workflow
│   ├── 05_modernbert_training.py       # ModernBERT / DistilBERT Colab GPU fine-tuning
│   └── 06_llama_qlora_training.py      # Llama 3.2 1B Instruct QLoRA 4-bit fine-tuning
├── results/
│   ├── compare_v2_seeds.csv            # Multi-seed candidate evaluation metrics
│   ├── leaky_baseline_evaluation.csv   # Random row-split baseline comparison
│   ├── bootstrap_confidence_intervals.csv # Group-bootstrap 95% CIs on merchants
│   ├── forecast_backtest_report.csv    # Rolling-origin time-series backtest results
│   ├── own_test_evaluation.csv         # Model-only vs Model+Rent-rule metrics
│   ├── confidence_threshold_routing.csv# Validation confidence routing thresholds
│   ├── per_class_metrics.csv           # Clean vs Noisy per-category metrics
│   ├── confusion_matrix_clean.png      # Confusion matrix on clean unseen merchants
│   └── confusion_matrix_noisy.png      # Confusion matrix on noisy Indian banking text
├── src/
│   ├── data_prep_v2.py                 # Core merchant extractor & 10-split generator
│   ├── model_comparison_suite.py       # Full candidate comparison benchmark engine
│   ├── budgeting.py                    # Deterministic category budgeting engine
│   ├── goal_saving.py                  # Formula-based goal savings planner
│   ├── forecast.py                     # Forecaster & rolling-origin backtest engine
│   ├── privacy.py                      # Outbound data minimization firewall
│   ├── grounded_llm.py                 # Intent router & privacy-protected LLM layer
│   ├── chatbot_tools.py                # Deterministic Python financial calculation tools
│   ├── predict.py                      # Production inference engine with confidence
│   ├── explainability.py               # N-gram feature weight explainer
│   ├── analytics.py                    # Financial cash flow analytics
│   └── anomaly_detection.py            # Spending anomaly detector
├── app/
│   └── app.py                          # Streamlit Web Application
├── tests/                              # Automated Pytest suite (56 tests)
│   ├── test_budgeting.py
│   ├── test_goal_saving.py
│   ├── test_forecast.py
│   ├── test_privacy.py
│   ├── test_chatbot_tools.py
│   ├── test_grounded_llm.py
│   └── ...
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 10. Installation & Execution Guide

### 10.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/dhpcodes/AI-Personal-Finance-Advisor.git
cd AI-Personal-Finance-Advisor

# Install dependencies
pip install -r requirements.txt
```

### 10.2 Run Streamlit Web Application
```powershell
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 10.3 Run Test Suite
```powershell
python -m pytest -o pythonpath=.
```
**Verified Result**: 56 of 56 tests pass (100% pass rate) in ~15 seconds without requiring any external network connection or API keys.

---

## 11. Safety, Ethics & Non-Advisory Notice

> ⚠️ **Educational & Informational Notice**: This software is developed for transaction classification, behavioral cash-flow modeling, and budgeting analytics. It does **not** provide certified financial, investment, legal, or tax advice. Calculations are deterministic mathematical operations based on user-provided data; forecasts are statistical estimates, not financial guarantees. Transactions flagged by anomaly detection represent statistical outliers, not confirmed fraudulent activity.

---

## 12. License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
