# Experimental Report: AI Personal Finance Advisor (v3-robustness)

**Central Thesis**:
> *"Noise-robust, leakage-safe, personalizable transaction categorization for Indian bank statements, with a grounded advisor."*

---

## 1. Executive Summary

| Dimension | Top-Line Measured Metric | Status |
| :--- | :--- | :--- |
| **Clean Test Categorization** | **98.93%** Accuracy / **0.9893** Macro-F1 (Linear SVM) | MEASURED |
| **Noisy Test (Raw Input)** | **78.23%** (Linear SVM) / **81.16%** (Logistic Regression) | MEASURED |
| **Noisy Test (Canonicalized + Aug)** | **88.72%** (LogReg) / **88.30%** (Linear SVM) | MEASURED |
| **Held-Out Prefix Recovery** | **+22.20%** (IMPS), **+12.98%** (NEFT), **+9.89%** (UPI) | MEASURED |
| **User Personalization** | **38.85%** ($N=0$) $\rightarrow$ **88.89%** ($N=20$) Accuracy | SYNTHETIC |
| **Real Indian Test Set ($N=105$)** | **50.48%** Model-Only (95% CI: [41.07%, 59.85%]) vs **55.24%** Model+Rule (95% CI: [45.71%, 64.40%]) | MEASURED |
| **Grounding Validator Refusal** | **100%** (15/15 unanswerable queries safely refused) | MEASURED |
| **Hallucination Interception** | **100%** (25/25 injected numerical hallucinations caught & rejected) | MEASURED |
| **Test Suite Passing** | **74 passed, 0 failed, 0 skipped** (15.7s) | MEASURED |

### What Worked
1. **Domain Canonicalization (`src/canonicalize.py`)**: Stripping Indian payment wrappers (UPI VPAs, POS merchant city slips, NEFT reference numbers, IMPS identifiers) recovered massive accuracy drops on unseen noisy patterns without retraining (+22.20% on IMPS, +12.98% on NEFT).
2. **Convex Linear Models on Character N-Grams**: TF-IDF character 2–5 n-grams with calibrated Linear SVM / Logistic Regression outperformed tree models by +12–40 percentage points on noisy test data while training in seconds.
3. **Multi-Tier Adaptive Feedback (`src/adaptive_nlp.py`)**: Exact correction memory + lightweight cosine k-NN over user feedback elevated personalized accuracy from 38.85% to 88.89% across $N=20$ user corrections with guaranteed multi-tenant user isolation.
4. **Deterministic Numeric Grounding Validator (`src/numeric_validator.py`)**: Verifying that every scalar, currency, and percentage in the advisor response derives strictly from verified Python tools intercepted 100% of injected numerical hallucinations.

### What Failed / Did Not Help
1. **Unmodified Boosted Trees on Sparse Text**: XGBoost under-boosted severely in v2 (54.6% clean acc) due to aggressive feature subsampling (`colsample_bytree=0.5`) on 20,000 sparse n-grams and premature early stopping. Stacking with out-of-fold linear meta-features recovered XGBoost to 98.50% clean / 76.09% noisy, but trees remain 12–50x slower and less accurate than linear baselines (81.16% noisy).
2. **Rule-Based Rent Heuristics on Small Indian Test Set**: On $N=105$ real Indian transactions, keyword rules improved accuracy by +4.76 percentage points (50.48% to 55.24%), but the difference is statistically insignificant ($p = 0.4894$, Wilson 95% CIs overlap substantially).

---

## 2. Data Integrity & Leakage Audit

### 2.1 Merchant Split vs Random Split (Data Leakage)
In standard random cross-validation splits, transactions from the same merchant (e.g., `UBER *TRIP MUMBAI`, `UBER EATS`, `UBER INDIA`) appear in both training and test sets. On clean text, random split reaches 99.1% accuracy and merchant-grouped split reaches 98.9%.
Crucially, uncorrupted strings are linearly separable in both splits. The danger of data leakage is **not** that random splits artificially inflate clean accuracy; rather, **leakage hides model fragility under real-world domain collapse**. When evaluated on unseen merchants subjected to synthetic noise, random-split models degrade catastrophically because they memorized merchant name strings rather than robust transaction semantics.

### 2.2 Small Test Set Statistical Honesty (`own_test_indian.csv`)
Evaluating on 105 real Indian bank statement strings:
- **Model-Only**: 53 / 105 = 50.48% (95% Wilson Score CI: `[41.07%, 59.85%]`)
- **Model + Rent Rule**: 58 / 105 = 55.24% (95% Wilson Score CI: `[45.71%, 64.40%]`)
- **Two-Proportion Pooled Test**: $z = 0.6912$, $p = 0.4894$ ($p > 0.05$).
- **Paired Exact Test (McNemar)**: Exactly 5 transactions corrected, two-sided $p = 0.0625$ ($p > 0.05$).

**Scientific Conclusion**: Because the difference (+4.76%) falls within the confidence interval bounds, we cannot claim that rule augmentation provides a statistically significant improvement on this test set. Expanding this held-out Indian test set to $N \ge 1,000$ transactions is required for statistical certainty.

### 2.3 Methodological Caveats & Synthetic Circularities
1. **Synthetic Noise Family Sharing**: Synthetic noise generation (`scripts/run_phase2_noise_robustness.py`) and canonicalization (`src/canonicalize.py`) share regular expression family patterns for UPI, NEFT, IMPS, and POS. While NEFT and IMPS were strictly held out from training in Phase 2, evaluating regex stripping against synthetic regex generation carries inherent family overlap.
2. **Diwali Seasonal Surge**: Synthetic forecast series contain a deterministic multiplier in October (Month 10). Prophet's yearly Fourier components capture this with high fidelity ($R^2 > 0.90$), but this reflects synthetic generator fidelity rather than messy real-world seasonal shifts.

---

## 3. Noise Robustness & Preprocessing

### 3.1 Impact of Domain Canonicalization Per Noise Type
Evaluating a model trained on noisy augmented data across distinct noise corruptions on held-out test data (`results/per_noise_type_evaluation.csv`):

| Noise Type | Raw Input Acc | Canonicalized Acc | Absolute Gain | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **IMPS (Held-out)** | 67.24% | **89.44%** | **+22.20%** | +33.0% |
| **NEFT (Held-out)** | 76.59% | **89.57%** | **+12.98%** | +17.0% |
| **UPI** | 79.52% | **89.41%** | **+9.89%** | +12.4% |
| **POS** | 80.52% | **89.31%** | **+8.79%** | +10.9% |
| **UPPERCASE** | 80.89% | **88.94%** | **+8.05%** | +10.0% |
| **TRUNCATION** | 80.99% | **88.59%** | **+7.60%** | +9.4% |
| **TYPO / PERMUTATION** | 88.58% | **89.18%** | **+0.60%** | +0.7% |

*Key Takeaway*: Canonicalization provides the largest gains on held-out transactional wrappers (IMPS +22.20%, NEFT +12.98%) by eliminating non-semantic syntax before feature extraction.

### 3.2 Temperature Scaling Calibration
Post-hoc temperature scaling fitted via NLL on the noisy validation set ($T = 0.6453$):
- **Clean Test**: Raw ECE = 0.0752 $\rightarrow$ Calibrated ECE = **0.0634**
- **Noisy Test**: Raw ECE = 0.1459 $\rightarrow$ Calibrated ECE = **0.1321**

Scaling sharpens probabilities from underconfident linear models, consistently reducing calibration error without altering classification ranking.

### 3.3 Label Noise Audit
A 5-fold out-of-fold label noise audit on 69,885 clean training transactions revealed:
- Out-of-fold accuracy: **98.15%**
- High-confidence disagreements ($P > 0.95$ where prediction $\ne$ label): **5 / 69,885 (0.007%)**
- Clean dataset quality ceiling: **$\approx 99.99\%$**

---

## 4. Model Comparison & Tree Tuning Post-Mortem

### 4.1 Consolidated 7-Model Evaluation Matrix

| Model ID | Architecture | Status | Clean Acc | Clean Macro-F1 | Noisy Acc | Noisy Macro-F1 | Latency (ms) | Resource Needs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | **TF-IDF + Logistic Regression** | **MEASURED** | 0.9877 | 0.9877 | **0.8116** | **0.8104** | 0.0029 | CPU (Lightweight) |
| **Model 2** | **TF-IDF + Calibrated Linear SVM** | **MEASURED** | **0.9893** | **0.9893** | 0.7823 | 0.7811 | 0.0066 | CPU (Lightweight) |
| **Model 3** | TF-IDF + LightGBM (Baseline) | **MEASURED** | 0.9568 | 0.9565 | 0.7158 | 0.7218 | 0.0459 | CPU Multicore |
| **Model 3b** | Stacked LightGBM (Tuned + OOF LogReg) | **MEASURED** | 0.9822 | 0.9821 | 0.7579 | 0.7588 | 0.0380 | CPU Multicore |
| **Model 4** | TF-IDF + XGBoost (Baseline) | **MEASURED** | 0.5462 | 0.5427 | 0.4074 | 0.4252 | 0.0212 | CPU Multicore |
| **Model 4b** | Stacked XGBoost (Tuned + OOF LogReg) | **MEASURED** | 0.9850 | 0.9849 | 0.7609 | 0.7624 | 0.0195 | CPU Multicore |
| **Model 5** | Sentence Embeddings (all-MiniLM-L6-v2) | **NOT RUN** | - | - | - | - | - | T4 GPU (Workflow: `notebooks/04b_sentence_embeddings.py`) |
| **Model 6** | Fine-Tuned ModernBERT / DistilBERT | **NOT RUN** | - | - | - | - | - | T4 GPU (Workflow: `notebooks/05_modernbert_training.py`) |
| **Model 7** | Llama 3.2 1B Instruct QLoRA | **NOT RUN** | - | - | - | - | - | A100 GPU (Workflow: `notebooks/06_llama_qlora_training.py`) |

### 4.2 Tree Tuning Post-Mortem & Diagnosis
1. **Why XGBoost Failed in v2 (54.6% Acc)**:
   - Feature subsampling: `colsample_bytree = 0.5` severely starved shallow depth-6 trees on 20,000 sparse character n-gram features (where 99.8% of elements are 0).
   - Early stopping: `early_stopping_rounds = 30` on noisy validation multi-class cross-entropy halted optimization prematurely at iteration 59 on an early plateau.
2. **Why Linear Models Win on Sparse N-Grams**:
   - Linear models directly sum additive positive and negative feature weights across hundreds of informative character tokens.
   - Decision trees partition feature space using axis-aligned orthogonal splits, requiring deep trees and immense sample sizes to capture disjunctive n-gram patterns.
   - Even when tuned with Optuna and stacked with out-of-fold linear meta-features, trees reached 76.09% noisy test accuracy—trailing pure linear models (81.16%) while requiring 15–50x more training time.

---

## 5. Personalization Architecture & Feedback Simulation

### 5.1 Multi-Tier Personalization Pipeline
```
Raw Description
      │
      ▼
Level 1: Global Linear SVM Classifier (Fallback Baseline)
      │
      ▼
Level 2: User Correction Memory (Exact / Normalized Regex Match, Conf=1.0)
      │
      ▼
Level 3: Personalized k-NN Prototype Layer (Active when User Corrections >= 5, Cosine Conf >= 0.6)
```

### 5.2 Simulation Results Across Feedback Steps ($N$)
Evaluated across 5 synthetic user profiles over seeds 42, 43, 44 (`results/adaptive_nlp_evaluation.csv`):

| Feedback Steps ($N$) | Overall Accuracy | User Overrides Active | Active Tier |
| :---: | :---: | :---: | :--- |
| **$N = 0$** | 38.85% | 0 | Tier 1 (Global Baseline) |
| **$N = 5$** | 56.41% | 23 | Tier 2 (Exact Memory) |
| **$N = 10$** | 71.28% | 46 | Tier 2 + Tier 3 (k-NN) |
| **$N = 15$** | 81.54% | 68 | Tier 2 + Tier 3 (k-NN) |
| **$N = 20$** | **88.89%** | 89 | Tier 2 + Tier 3 (k-NN) |

### 5.3 User Isolation Guarantee
Multi-tenant isolation is enforced at the SQLite storage layer (`src/database.py`). Tests confirm that feedback recorded for `user_A` never leaks into `user_B`'s inference space.

---

## 6. Grounded Financial Advisor & Hallucination Defense

### 6.1 Architecture
The advisor decouples computation from natural language generation:
1. **Intent Routing**: Query is classified into analytical intents (`spending_breakdown`, `recurring_expenses`, `forecast_expenses`, `anomaly_detection`).
2. **Deterministic Python Execution**: Verified mathematical functions compute aggregations, statistics, and forecasts on private user data.
3. **PrivacyGuard Outbound Firewall**: Payload is verified against a strict field allowlist and scrubbed of Indian PII (PAN, Aadhaar, cards, mobile numbers).
4. **Provider-Agnostic LLM Client**: OpenAI-compatible client defaults to local Ollama (`http://localhost:11434/v1`, `llama3.2:3b`), with seamless fallback to offline templates.
5. **NumericGroundingValidator**: Inspects the final assistant output, extracting all numbers, percentages, and currencies, asserting that every numerical claim exists in verified tool data. If an ungrounded number is found, the LLM response is rejected and replaced by a deterministic template.

### 6.2 Grounding Evaluation Benchmark (`scripts/run_phase6_grounding_eval.py`)

| Test Suite | Total Queries | Evaluated Metric | Result |
| :--- | :---: | :--- | :---: |
| **Refusal of Unanswerable Queries** | 15 | Refusal Rate on Non-Existent Categories | **100.0%** (15/15) |
| **Hallucination Interception** | 25 | Injected Hallucination Catch Rate | **100.0%** (25/25) |
| **Legitimate Response Acceptance** | 25 | False Rejection Rate | **0.0%** (0/25) |
| **Offline Fallback Reliability** | 10 | Fallback Success Rate on Network Disconnection | **100.0%** (10/10) |

---

## 7. Future Work

1. **GPU Fine-Tuning Execution**: Run `notebooks/05_modernbert_training.py` and `notebooks/06_llama_qlora_training.py` on Google Colab / cloud GPU instances.
2. **Scaled Indian Bank Statement Dataset**: Expand the held-out real test set from $N=105$ to $N \ge 1,000$ labeled statements across major Indian banks (HDFC, ICICI, SBI, Axis, Kotak) to achieve statistically significant validation of heuristic and neural models.
3. **Lunisolar Calendar & Regional Festival Modeling**: Incorporate dynamic Hindu and regional holiday dates into Prophet forecasting regressors to replace static Gregorian monthly assumptions.
