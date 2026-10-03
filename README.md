# AI Personal Finance Advisor

A production-grade, modular, testable, and explainable **AI Personal Finance Advisor** built with Python, Scikit-Learn, PyArrow, pandas, and Streamlit.

The application automatically categorizes financial transaction descriptions, audits data leakage, performs financial cash flow analytics, detects unusual spending patterns, provides n-gram model explainability, features an interactive **AI Financial Chatbot**, and serves a web dashboard.

---

## 1. Project Overview & Problem Statement

Categorizing raw financial transactions (e.g., `"Starbucks Store #8831"`, `"Salary Direct Deposit"`, `"UCLA Medical Center #1029"`) into meaningful spending categories is a foundational requirement for personal finance tracking, budget allocation, and cash flow analysis.

This project delivers an end-to-end Machine Learning pipeline and interactive web application:
- **Automatic Classification**: Classifies transaction text into 10 standardized finance categories using a calibrated Linear SVM (98.69% accuracy).
- **AI Financial Chatbot**: Interactive conversational interface using deterministic Python financial tools grounded with Google Gemini LLM formatting (with optional support for OpenAI / Anthropic) and offline fallback execution.
- **Data Leakage Auditing**: Quantifies and prevents train/test text memorization (0.00% leakage achieved).
- **Financial Analytics**: Aggregates cash flow, income vs. expenses, category breakdown %, and monthly timeline trends.
- **Spending Anomaly Detection**: Identifies statistical outliers using `IsolationForest` and z-score thresholds (>2.5 std dev) formatted dynamically in dataset currency.
- **Model Explainability**: Highlights specific words/n-grams driving classification decisions.

---

## 2. AI Financial Chatbot Architecture & Grounding Principle

The chatbot follows a strict **modular, tool-based, deterministic architecture**:

```
                       USER QUESTION
                            │
                            ▼
    STREAMLIT CHAT INTERFACE (app/app.py - st.chat_input & st.session_state)
                            │
                            ▼
    INTENT ROUTER & CONTEXT MANAGER (src/grounded_llm.py)
                            │
                            ▼
    DETERMINISTIC FINANCIAL TOOLS (src/chatbot_tools.py)
      ├── get_spending_by_category()
      ├── get_category_spending_for_period()
      ├── get_recurring_expenses()
      ├── explain_transaction_classification()
      ├── forecast_monthly_expenses()
      ├── get_monthly_income_expense()
      ├── get_transaction_details()
      └── detect_unusual_spending()
                            │
                            ▼
    CALCULATION & EVIDENCE LAYER (Executed on user's active DataFrame)
                            │
                            ▼
    GROUNDED LLM LAYER (Google Gemini API / OpenAI / Anthropic / Offline Fallback)
                            │
                            ▼
    RESPONSE & EVIDENCE DISPLAY (Answer + Details + Period + Method + Limitations)
```

### Core Design Principles:
1. **Deterministic Calculations**: All financial totals, percentages, balances, recurring patterns, and forecasts are calculated strictly by Python tools.
2. **Zero Financial Hallucination**: The LLM is restricted from inventing transaction records, balances, or forecasts.
3. **Offline Fallback Resilience**: If no API key is set or the network fails, the chatbot automatically uses an offline natural-language template generator without crashing.
4. **Data Privacy**: Raw transaction datasets are processed locally in session state; only pre-computed numeric summaries are sent to the LLM interface.

---

## 3. Supported Chatbot Questions

The chatbot natively interprets and answers queries such as:
1. *"Where am I spending the most money?"*
2. *"How much did I spend on food this month?"*
3. *"What are my recurring expenses?"*
4. *"Why was 'Netflix Subscription' classified as entertainment?"*
5. *"What could my expenses look like next month?"*
6. *"Show my monthly income versus expenses."*
7. *"Are there any unusual spending transactions?"*

---

## 4. Empirical Model Benchmarks & Data Leakage Prevention

The benchmark suite evaluated candidate classifiers on a **30,000 holdout test set**:

| Candidate Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** 🏆 | **`0.9869`** | **`0.9869`** | **`0.9869`** | **`0.9869`** | **`0.9869`** |
| **Logistic Regression** | `0.9855` | `0.9855` | `0.9855` | `0.9855` | `0.9855` |
| **Multinomial Naive Bayes** | `0.9697` | `0.9702` | `0.9697` | `0.9697` | `0.9697` |

- **Data Leakage Audit**: Standard random splits exhibited **74.69%** text overlap. This pipeline implements deduplicated text splitting to guarantee **0.00% text leakage**.

---

## 5. LLM API Provider Configuration & Secrets

### Supported LLM Providers & Models:
- **Google Gemini (Default)**: Model `gemini-2.5-flash` or `gemini-1.5-flash` via official `google-genai` SDK.
- **OpenAI**: Model `gpt-4o-mini` or `gpt-4o` via `openai` SDK.
- **Anthropic**: Model `claude-3-5-haiku-20241022` via `anthropic` SDK.

### How to Obtain an API Key:
- **Google Gemini**: Obtain a key from [Google AI Studio](https://aistudio.google.com/).
- **OpenAI**: Obtain a key from [OpenAI Platform](https://platform.openai.com/).
- **Anthropic**: Obtain a key from [Anthropic Console](https://console.anthropic.com/).

### Environment Variable Configuration (`.env`):
Copy `.env.example` to `.env` locally:
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
LLM_TIMEOUT_SECONDS=30
LLM_MAX_OUTPUT_TOKENS=1000

# Set the key for your selected provider:
GEMINI_API_KEY=your_gemini_api_key_here
```

### Streamlit Community Cloud Secrets Configuration:
In your Streamlit Cloud Dashboard under **App Settings -> Secrets**:
```toml
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

### Configuration Precedence:
1. `st.secrets` (Streamlit Deployment)
2. `.env` file (Local Development)
3. Process Environment Variables (`os.getenv`)
4. Offline Deterministic Fallback Mode (if no key is provided)

---

## 6. System Architecture & Repository Structure

```
AI-Personal-Finance-Advisor/
├── configs/
│   └── config.yaml             # Configuration parameters
├── data/
│   ├── raw/0000.parquet        # Raw HuggingFace Parquet dataset
│   └── processed/              # Data leakage-safe train/val/test splits
├── models/
│   └── final_model.joblib      # Winning Linear SVM serialized artifact
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
│   ├── anomaly_detection.py    # IsolationForest anomaly detector with dynamic currency formatting
│   ├── explainability.py       # N-gram feature weight explainer
│   ├── database.py             # SQLite persistence layer with user isolation
│   ├── adaptive_nlp.py         # Multi-level personalized NLP engine
│   ├── behavioral_engine.py    # User spending profile extractor
│   ├── chatbot_tools.py        # Deterministic Python financial calculation tools
│   └── grounded_llm.py         # Provider abstraction, intent router, Gemini/OpenAI/Anthropic & offline template layer
├── app/
│   └── app.py                  # Streamlit Web Dashboard with Chatbot Navigation
├── tests/                      # Automated Pytest unit test suite
│   ├── test_analytics.py
│   ├── test_data_validation.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   ├── test_chatbot_tools.py
│   ├── test_grounded_llm.py
│   └── test_llm_integration.py
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

---

## 7. Installation & Execution Workflow

### 1. Installation
```bash
git clone https://github.com/dhpcodes/AI-Personal-Finance-Advisor.git
cd AI-Personal-Finance-Advisor
pip install -r requirements.txt
```

### 2. Run Streamlit Application (Windows PowerShell)
```powershell
streamlit run app/app.py
```

### 3. Run Automated Test Suite (Without requiring an API key)
```powershell
python -m pytest tests/
```

---

## 8. Common API Troubleshooting & Privacy Notes

- **Missing API Key**: The app displays an informative banner and automatically uses offline template generation. All financial calculations remain 100% operational.
- **API Timeouts / Rate Limits**: The LLM client handles timeouts via `LLM_TIMEOUT_SECONDS` (default 30s). On rate limits or network failures, the assistant logs a warning and falls back to offline answers.
- **Data Privacy**: Only pre-calculated numeric summaries and category names are sent to external LLM endpoints. No bank account numbers, passwords, or raw user databases are logged or transmitted.

---

## 9. Financial Advice Safety Disclaimer

> ⚠️ **Educational Purpose Only**: This application is an educational financial management tool. It does **not** provide certified financial, legal, investment, or tax advice. Always consult a licensed professional for formal financial accounting.

---

## 10. License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
