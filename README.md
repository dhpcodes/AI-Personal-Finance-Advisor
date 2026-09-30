# AI Personal Finance Advisor

A production-grade, modular, testable, and explainable **AI Personal Finance Advisor** built with Python, Scikit-Learn, PyArrow, pandas, and Streamlit.

The application automatically categorizes financial transaction descriptions, audits data leakage, performs financial cash flow analytics, detects unusual spending patterns, provides n-gram model explainability, and serves an interactive web dashboard.

---

## 1. Project Overview & Problem Statement

Categorizing raw financial transactions (e.g., `"Starbucks Store #8831"`, `"Salary Direct Deposit"`, `"UCLA Medical Center #1029"`) into meaningful spending categories is a foundational requirement for personal finance tracking, budget allocation, and cash flow analysis.

This project delivers an end-to-end Machine Learning pipeline and web application designed to solve this problem cleanly:
- **Automatic Classification**: Classifies transaction text into 10 standardized finance categories.
- **Data Leakage Auditing**: Explicitly quantifies and prevents train/test text memorization.
- **Financial Analytics**: Aggregates cash flow, income vs. expenses, category breakdown %, and monthly timeline trends.
- **Spending Anomaly Detection**: Identifies statistical outliers using `IsolationForest` and z-score thresholds.
- **Explainability**: Highlights specific words/n-grams that drove the classification decision.

---

## 2. Verified Raw Dataset Inspection

All metrics below are **VERIFIED FROM RAW DATA** (`data/raw/0000.parquet` from `mitulshah/transaction-categorization`):

| Metric | Verified Value |
| :--- | :--- |
| **Total Transaction Rows** | `4,501,043` |
| **Columns** | `transaction_description`, `category`, `country`, `currency` |
| **Missing Values** | `0` (0.00%) |
| **Unique Descriptions** | `1,387,044` |
| **Categories Count** | `10` |
| **Exact Duplicate Rows** | `2,940,825` (65.33%) |
| **Descriptions Mapped to >1 Category** | `2,485` |

### Category Distribution
1. **Utilities & Services**: `451,842` (10.04%)
2. **Government & Legal**: `451,108` (10.02%)
3. **Financial Services**: `450,959` (10.02%)
4. **Income**: `450,545` (10.01%)
5. **Charity & Donations**: `450,133` (10.00%)
6. **Shopping & Retail**: `449,941` (10.00%)
7. **Healthcare & Medical**: `449,857` (9.99%)
8. **Entertainment & Recreation**: `449,495` (9.99%)
9. **Transportation**: `449,235` (9.98%)
10. **Food & Dining**: `447,928` (9.95%)

### Country & Currency Breakdown (1:1 Mapping)
- **Australia (AUD)**: `901,765`
- **India (INR)**: `901,544`
- **United Kingdom (GBP)**: `899,915`
- **United States (USD)**: `899,163`
- **Canada (CAD)**: `898,656`

---

## 3. Data Leakage Prevention & Audit Results

In transaction classification datasets, high frequency of repeated merchant names (e.g. `"Uber Ride"`, `"Amazon Store"`) creates severe data leakage in standard random splits.

- **Random Stratified Split Leakage Audit**: **`74.69%`** of test transactions shared identical descriptions with the training set.
- **Mitigation Strategy**: The pipeline implements deduplicated stratified splitting, grouping by unique transaction text to guarantee **`0.00%`** text leakage between train, validation, and test splits.

---

## 4. Empirical Model Evaluation Benchmarks

The benchmark suite evaluated 3 candidate classifiers on a **30,000 holdout test set** (TF-IDF word n-grams `(1,2)` fitted **ONLY** on the training split):

| Candidate Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** 🏆 | **`0.9869`** | **`0.9869`** | **`0.9869`** | **`0.9869`** | **`0.9869`** | `24.134s` |
| **Logistic Regression** | `0.9855` | `0.9855` | `0.9855` | `0.9855` | `0.9855` | `4.511s` |
| **Multinomial Naive Bayes** | `0.9697` | `0.9702` | `0.9697` | `0.9697` | `0.9697` | **`0.036s`** |

> **Selected Final Model**: **Linear SVM** with sigmoid probability calibration (`models/final_model.joblib`).

---

## 5. System Architecture

```
AI-Personal-Finance-Advisor/
├── configs/
│   └── config.yaml             # Configuration parameters
├── data/
│   ├── raw/0000.parquet        # Raw HuggingFace Parquet dataset
│   └── processed/              # Data leakage-safe train/val/test splits
├── notebooks/                  # EDA & training notebooks
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── config.py               # Settings manager
│   ├── data_loader.py          # PyArrow/pandas data reader
│   ├── data_validation.py      # Schema & upload validation
│   ├── preprocessing.py       # Text cleaner & tokenizer
│   ├── split_data.py           # Reproducible data splitter & leakage auditor
│   ├── train.py                # Classifier training pipeline
│   ├── evaluate.py             # Benchmark & confusion matrix generator
│   ├── predict.py              # Inference predictor with confidence
│   ├── analytics.py            # Personal finance cash flow analytics
│   ├── anomaly_detection.py    # IsolationForest anomaly detector
│   └── explainability.py       # N-gram feature weight explainer
├── models/
│   └── final_model.joblib      # Winning Linear SVM serialized artifact
├── reports/                    # Generated inspection & evaluation CSV reports
├── app/
│   └── app.py                  # Interactive Streamlit Web Dashboard
├── tests/                      # Automated Pytest test suite
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

---

## 6. Installation & Setup

### Requirements
- Python 3.9+
- Recommended OS: Windows, Linux, or macOS

### Environment Setup
```bash
# 1. Clone or navigate to the repository
cd AI-Personal-Finance-Advisor

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 7. Execution Workflow & Pipeline Commands

### Phase 2: Dataset Inspection & Report Generation
```bash
python -m src.run_phase2
```

### Phase 4: Data Leakage Audit & Train/Val/Test Split
```bash
python -m src.run_phase4
```

### Phases 5-7: Model Training, Evaluation & Final Model Selection
```bash
python -m src.run_phases5_6_7
```

### Run Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```

### Run Automated Unit Tests
```bash
python -m pytest tests/
```

---

## 8. Financial Advice Safety & Privacy Disclaimers

### Privacy & Security
- **Local Processing**: Uploaded financial transaction files are processed strictly within your local application session.
- **No Private Data Logging**: No raw bank account numbers, passwords, or personal credentials are saved to external servers or public Git repositories.

### Safety Disclaimer
> ⚠️ **Educational Purpose Only**: This application is an educational financial tool providing automated transaction categorization and statistical spending insights. It does **not** provide certified financial, legal, investment, or tax advice. Always consult a licensed professional for formal financial accounting.

---

## 9. License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
