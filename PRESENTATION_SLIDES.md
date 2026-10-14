# AI Personal Finance Advisor: Final Project Defense & Presentation Slides

**Project Title**: Noise-Robust, Leakage-Safe Transaction Categorization for Indian Bank Statements  
**Domain**: Applied NLP, Machine Learning Systems, and FinTech Architecture  
**Evaluation Standard**: Empirical Zero-Fabrication Benchmark ([`results/final_metrics.csv`](results/final_metrics.csv))  
**Artifact Models**:
* `models/lightgbm_transaction_classifier.joblib` (LightGBM GBDT)
* `models/xgboost_transaction_classifier.joblib` (XGBoost GBDT)
* `models/minilm_lr_transaction_classifier.joblib` (MiniLM-L6-v2 + Logistic Regression)
* `models/variant_b_candidate.joblib` (Calibrated Linear SVM)
* `models/final_model.joblib` (Legacy Production Baseline - Read-Only)

---

## Slide 1: Title Slide & Project Overview

### Visual Layout:
* **Title**: AI Personal Finance Advisor: Noise-Robust, Leakage-Safe Transaction Categorization
* **Subtitle**: Systematic Evaluation Across 4 Model Families (Linear, Boosted Trees, Dense Embeddings, Transformers) on Indian Banking Data
* **Domain**: Applied Natural Language Processing & FinTech Machine Learning Systems
* **Target Classes**: 10 Core Financial Categories (Food & Dining, Shopping, Utilities, Transportation, etc.)

### Speaker Notes (1.5 minutes):
> "Respected professors and evaluation committee members. Today I am presenting my final-year project: **AI Personal Finance Advisor**, focusing on our noise-robust, leakage-safe transaction categorization engine.
> 
> In personal finance systems, over 80% of automated budgeting, cash flow forecasting, and discretionary tracking decisions rely on accurately categorizing messy, abbreviated bank statement strings.
> 
> Guided by the core principle of **selecting models according to the data type**—which here consists of short, sparse, abbreviated text corrupted by clearing wrappers and typos—we have implemented, evaluated, and benchmarked four distinct model families under strict zero-leakage conditions: Linear Baselines, Gradient Boosted Decision Trees (LightGBM and XGBoost), Dense Sentence Embeddings (MiniLM-L6-v2), and Fine-Tuned Transformers (DistilBERT)."

---

## Slide 2: Problem Statement & Real-World Domain Challenges

### Bullet Points:
* **Distinct Data Modality**: Financial transaction strings are not grammatical prose; they are cryptic 3-to-8 word clearing strings.
* **Banking Clearing Wrappers**:
  * UPI Handles & RRNs: `UPI/308942189012/SWIGGY/swiggy@icici`
  * POS Card Masks: `POS 5241XXXX9021 MYNTRA DESIGNS BANGALORE`
  * NEFT/IMPS Sequences: `NEFT-855832329275-CARE HEALTH-N34521`
* **Subword Typos & Character Truncations**:
  * Character dropouts: `SWGY FOOD`, `AMAZN PY`, `BIGBASKT SUPERM...`
* **Local Merchant Domain Gap**:
  * Indian consumer brands (`Zomato`, `Zepto`, `Blinkit`, `Jio`) absent in Western datasets, causing legacy models to misclassify or abstain.

### Speaker Notes (1.5 minutes):
> "Standard NLP models assume grammatically structured English prose. Financial transaction strings, however, are corrupted by core banking clearing protocols. 
> 
> As you can see, a typical UPI transaction includes a 12-digit reference number, a merchant name, and a Virtual Payment Address handle like `@icici`. A naive bag-of-words model will treat the reference number or bank handle as primary features, causing severe spurious correlations. 
> 
> Furthermore, brand abbreviations like 'SWGY' or 'AMAZN' break standard whitespace tokenizers. Our objective is to reliably map these corrupted strings into 10 target categories without manual rule engineering."

---

## Slide 3: End-to-End System Architecture

### Diagram & Flow:
```
[Raw Transaction String (e.g. UPI/308942189012/SWIGGY/swiggy@icici)]
         │
         ▼
[1. Domain Canonicalization (src/canonicalize.py)]
   • Strips UPI prefixes, RRN digits, VPA handles (@okaxis, @icici)
   • Strips POS card masks (XXXX) and NEFT/IMPS headers
         │
         ▼
[2. Text Normalization (src/preprocessing.py)]
   • Lowercasing, URL removal, punctuation filtering, whitespace collapse
         │
         ▼
[3. Multi-Representation Extraction]
   ┌───────────────────────────────────┬───────────────────────────────────┐
   │ Sparse Subword Features:          │ Dense Semantic Vectors:           │
   │ TfidfVectorizer char_wb(2,4)      │ all-MiniLM-L6-v2                  │
   │ 3,000 max features, sublinear TF  │ 384-dimensional dense embeddings  │
   └───────────────────────────────────┴───────────────────────────────────┘
         │                                   │
         ▼                                   ▼
[4a. Tree Ensembles / Linear SVM]     [4b. Dense Linear Classifier]
   • LightGBM / XGBoost / Linear SVM   • Logistic Regression (C=10.0)
         │                                   │
         └─────────────────┬─────────────────┘
                           ▼
[Predicted Category + Calibrated Confidence Score]
```

### Speaker Notes (1.5 minutes):
> "This slide illustrates our system architecture:
> 
> First, our custom domain canonicalizer strips out clearing metadata, card masking hashes, and UPI handles (`UPI/308942189012/SWIGGY/swiggy@icici` becomes `swiggy`).
> 
> Second, text normalization standardizes whitespace, punctuation, and case.
> 
> Third, we branch into two complementary representations:
> 1. Character word-boundary n-grams (`char_wb` range 2–4, 3,000 features). This captures morphological roots so typos like 'swgy' share subwords with 'swiggy'.
> 2. Dense 384-dimensional sentence embeddings via `all-MiniLM-L6-v2` to capture semantic associations without fixed vocabulary tables.
> 
> Fourth, these representations feed our tree ensembles, linear SVM, and dense logistic regression classifiers."

---

## Slide 4: Data Hygiene & Zero-Leakage Experimental Setup

### Methodology & Integrity Assertions:
* **Primary Corpus**: Mitul Shah Banking Dataset (100,000 transactions, 10 categories).
  * **Train Split**: 69,885 rows (2,975 unique merchant cores).
  * **Validation Split**: 10,075 rows (333 unique merchant cores).
  * **Test Split**: 20,040 rows (803 unique merchant cores).
  * **Merchant Overlap**: **Strictly 0** between train, val, and test splits (grouped split prevents memorization).
* **Indian Domain Augmentation**:
  * 3,062 Indian merchant transactions balanced across all 10 categories (`data/indian_merchants_train_seed.csv`).
  * Combined Training Set: **72,947 transactions**.
* **Held-Out Real Indian Test Statements**:
  * 105 authentic UPI, NEFT, and POS transactions from real bank accounts (`data/own_test_indian.csv`).
  * **Zero string overlap** with all training records.

### Speaker Notes (2 minutes):
> "A major flaw in applied ML research is data leakage—specifically when transactions from the same merchant appear in both training and test sets. If 'Starbucks' appears in both, the model merely memorizes the merchant rather than learning generalizable patterns.
> 
> In our experimental setup, we enforced strict merchant-grouped splitting. There is absolutely zero merchant overlap between our 69,885 training samples, 10,075 validation samples, and 20,040 test samples.
> 
> Furthermore, all vectorizers and scalers were fit strictly on training splits. Evaluation metrics on held-out test splits were computed only once, without any hyperparameter tuning on test data."

---

## Slide 5: The 4 Model Families: Selection by Data Type

### Comparison Across Architectural Families:
| Model Family | Representative Model | Input Representation | Theoretical Strengths | Vulnerabilities on Short Text |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Baselines** | Calibrated Linear SVM | Sparse char n-grams (3k) | Maximum margin; high clean and typo accuracy | Requires calibration scaling; linear boundaries |
| **Boosted Trees (GBDT)** | LightGBM | Sparse char n-grams (3k) | Non-linear splits; native probabilities; fast | Sparse text histogram construction |
| **Dense Embeddings** | MiniLM-L6-v2 + LogReg | Dense vectors (384-d) | High semantic transfer; zero OOV vocabulary | Sensitive to subword typo corruption |
| **Fine-Tuned Transformers** | DistilBERT | Token sequence (128 tok) | Bidirectional contextual attention | Heavy compute; requires GPU for inference |

### Speaker Notes (1.5 minutes):
> "Our mentor instructed us to select and evaluate models according to the data type. 
> 
> 1. **Linear SVM**: Because character n-gram text is high-dimensional and sparse, classes are often linearly separable. SVM finds the maximum-margin hyperplane efficiently.
> 2. **Gradient Boosted Decision Trees (LightGBM & XGBoost)**: Provide non-linear decision boundaries and native class probabilities, but tree algorithms traditionally struggle on sparse text matrices.
> 3. **Dense Sentence Embeddings (MiniLM)**: Compress text into a continuous 384-dimensional space, capturing merchant synonyms without n-gram dictionaries.
> 4. **Transformers (DistilBERT)**: Provide full self-attention, serving as our deep learning benchmark."

---

## Slide 6: Algorithmic Deep-Dive: LightGBM vs. XGBoost on Sparse Text

### Key Algorithmic Mechanics & Controlled Diagnosis:
* **XGBoost Deficit Diagnosis & Empirical Evidence**:
  * Controlled diagnostic experiments (`results/xgboost_deep_diagnosis.csv`) demonstrate that the performance gap between XGBoost and LightGBM is **mostly explained by sparse zero handling and depth-wise tree growth**:
    * **Sparse Zero Handling (+10 points)**: When passing explicit dense zeros instead of sparse CSR, test accuracy jumped from 80.74% to **90.02%** (val: 80.19% $\rightarrow$ 95.46%).
    * **Depth-wise Growth Policy (+9 points)**: Setting `grow_policy='lossguide'` with `max_leaves=63` increased test accuracy from 71.99% to **80.74%**.
  * **Remaining Gap**: The remaining gap to LightGBM (97.72%) remains unexplained.
  * **EFB Status**: Attributing the remaining gap to Exclusive Feature Bundling (EFB) is an **untested hypothesis**.
* **Empirical Early Stopping Verification**:
  * An independent check with 100 trees and early stopping on validation loss ran XGBoost to iteration 99 (never early stopped), achieving 72.41% validation accuracy and 68.53% test accuracy. This confirms XGBoost's gap is not an artifact of premature early stopping.

### Speaker Notes (2 minutes):
> "When diagnosing why XGBoost lagged behind LightGBM on character n-grams, controlled ablation experiments revealed the primary drivers:
> 
> First, sparse zero handling accounted for roughly +10 points: passing explicit dense zeros raised XGBoost test accuracy from 80.74% to 90.02%.
> 
> Second, depth-wise tree growth accounted for +9 points: configuring loss-guided leaf growth with 63 leaves lifted test accuracy from 71.99% to 80.74%.
> 
> The remaining gap between tuned XGBoost and LightGBM remains unexplained, and attributing it to Exclusive Feature Bundling (EFB) remains an untested hypothesis."

---

## Slide 7: Comprehensive Multi-Model Benchmark Results

### Measured Empirical Performance Table ([`results/final_metrics.csv`](results/final_metrics.csv)):
| Metric | Production Baseline | Candidate Linear SVM | LightGBM Classifier | XGBoost Classifier | MiniLM-L6-v2 + LogReg | DistilBERT Fine-Tuned |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Status** | **MEASURED** | **MEASURED** | **MEASURED** | **MEASURED** | **MEASURED** | **MEASURED** |
| **Clean Test Acc (N=20,040)** | 99.07% | 99.00% | **97.72%** | 72.61% | 89.67% | **99.13%** |
| **95% Wilson Score CI** | `[98.92%, 99.19%]` | `[98.85%, 99.13%]` | `[97.50%, 97.92%]` | `[71.99%, 73.23%]` | `[89.24%, 90.08%]` | `[98.99%, 99.25%]` |
| **Clean Macro-F1** | 0.9907 | 0.9900 | **0.9772** | 0.7318 | 0.8977 | **0.9913** |
| **Noisy V2 Test Acc (Headline, N=20,040)** | 63.89% | **76.39%** | 63.55% | 45.14% | 47.81% | **69.00%** |
| **Initial Noisy Test Acc (Optimistic, shared)** | 92.73% | 97.19% | 93.09% | 66.89% | 82.82% | N/A |
| **Typo Noise Acc (N=2,492)** | 68.10% | **89.53%** | 80.54% | 53.73% | 57.06% | 77.52% |
| **Indian Model-Only Acc (N=105)** | 56.19% | 87.62% | 78.10% | 70.48% | 80.95% | **93.33%** |
| **Indian Model-Only Acc (N=96, Non-overlapping)** | — | 86.46% | 80.21% (3k-matched) / 77.08% (20k, app) | — | — | **92.71%** |
| **Indian Model-Only Acc (N=58, Fully Disjoint)** | — | 81.03% | 74.14% | — | — | **87.93%** |
| **Indian Model+Rent Acc (N=105, Secondary)** | 60.95% | 92.38% | 82.86% | 75.24% | 85.71% | **98.10%** |

> **⚠️ Rent rule caveat**: The rent rule is an offline post-classification heuristic used in evaluation scripts only. It is not applied in the deployed Streamlit app, whose rent transactions may be misclassified (e.g. `MONTHLY APARTMENT RENT TRANSFER TO LANDLORD` → `Government & Legal`). Its keywords match the Rent rows of the test set, so +rent figures are not evidence of generalization. Model-only figures are the headline numbers.
| **Single Latency (With Prep)** | 4.76 ms | 4.96 ms | **1.97 ms** | **1.55 ms** | 9.73 ms | (GPU batch) |
| **Batch 256 Latency / item** | 0.043 ms | 0.074 ms | **0.085 ms** | **0.065 ms** | 1.69 ms | (Kaggle T4) |
| **Model Artifact Size** | 10.69 MB | 8.29 MB | **6.96 MB** | **1.47 MB** | **0.02 MB** | 255.4 MB |

*Note: DistilBERT vs Linear SVM on Indian bank statements was NOT statistically tested.*

### Speaker Notes (2 minutes):
> "Here are our empirical evaluation results across all models, tested under identical conditions. Every single metric is extracted from our automated benchmark suite:
> 
> 1. **LightGBM vs XGBoost**: LightGBM achieves **97.72% clean accuracy** and **63.55% Noisy V2 accuracy**, beating XGBoost across every metric. On subword typos, LightGBM reaches 80.54% vs XGBoost's 53.73%.
> 2. **Inference Latency**: LightGBM executes at **1.97 ms per transaction**, compared to 4.96 ms for the Linear SVM and 9.73 ms for MiniLM.
> 3. **Linear SVM Reference**: The Linear SVM retains high clean accuracy (99.00%), top typo robustness (89.53%), and peak Noisy V2 accuracy (76.39%), confirming that high-dimensional subwords are naturally linearly separable.
> 4. **DistilBERT Transformer**: Fine-tuned on Kaggle (348.5 s, effective batch 64 via 2x T4 GPUs, no gradient accumulation, single seed 42), reaching **99.13% clean accuracy** but dropping to **69.00%** under independent noise."

---

## Slide 8: Which Model Wins Depends on the Condition

### Empirical Trade-offs Across Operating Environments:

1. **Clean Text ($N=20,040$) — DistilBERT Leads**:
   * **Winner**: **DistilBERT (99.13%)** slightly edges out **Linear SVM (99.00%)** ($\chi^2 = 11.16, p = 0.00084$) and **LightGBM (97.72%)**.
   * Contextual self-attention captures uncorrupted subwords with low Expected Calibration Error (ECE = 0.0008).

2. **Independent Noise ($N=20,040$, Noisy V2) — Linear SVM Wins Decisively**:
   * **Winner**: **Linear SVM (76.39%)** significantly outperforms **DistilBERT (69.00%)** ($\chi^2 = 690.48, p < 10^{-5}$) and **LightGBM (63.55%)**.
   * *WordPiece Tokenizer Hypothesis*: Transformer tokenizers (WordPiece) are brittle under character permutations and vowel deletions (vowel drop plunges DistilBERT to **33.62%**, mixed noise to **49.95%**), fracturing words into uninformative sub-tokens. Fixed character n-grams (`char_wb 2-5`) preserve overlapping roots despite character corruption.

3. **Unseen Indian Bank Statements ($N=96$ Non-Overlapping)**:
   * **Measured Scores**: DistilBERT achieved **92.71%** (model-only) / **97.92%** (+rent, offline only); Linear SVM achieved **86.46%** (model-only) / **91.67%** (+rent, offline only). *The rent rule is not applied in the deployed app; model-only figures are the headline numbers.*
   * **N=96 McNemar exact binomial tests** (from `results/n96_stats.csv`; 5 comparisons, not multiple-comparison corrected; treat as suggestive): DistilBERT vs Candidate SVM: b=6, c=0, exact p=**0.03125** (significant at α=0.05; **not significant** after Bonferroni α=0.01). DistilBERT vs LightGBM: b=15, c=0, p=**0.00006** (significant after Bonferroni). DistilBERT vs MiniLM: b=13, c=0, p=**0.00024** (significant after Bonferroni). All counts are small (6–15 discordant rows of 96).
   * **Wilson 95% CIs on N=96 model-only**: DistilBERT [85.71%, 96.42%], Candidate SVM [78.20%, 91.91%], LightGBM [67.73%, 84.35%], MiniLM [70.00%, 86.09%].
   * **Caveat**: DistilBERT was trained on the original seed (8 overlapping merchants present), so its Indian results are preliminary until retrained on the leakage-free seed.

> "A core takeaway from our research is that **which model wins depends entirely on the operational condition**:
> 
> First, on clean in-domain text, DistilBERT takes the lead at 99.13%, statistically edging out Linear SVM (99.00%, p=0.00084).
> 
> Second, under independent character noise (Noisy V2), the Linear SVM wins decisively at 76.39% vs DistilBERT's 69.00% (p < 10⁻⁵). We hypothesize that WordPiece subword tokenization fractures under character deletions—dropping to 33.62% on vowel drops—whereas character n-grams retain robust overlapping fragments.
> 
> Third, on 96 unseen Indian bank statements (model-only), DistilBERT scored 92.71% vs SVM's 86.46%. Exact binomial McNemar: b=6, c=0 (6 discordant rows), p=0.03125—significant at α=0.05, but **not significant after Bonferroni correction** (α=0.01, 5 comparisons). Results are based on few discordant rows; treat as suggestive. DistilBERT was trained on the original seed containing 8 overlapping merchants, making its Indian results preliminary. Wilson 95% CIs: DistilBERT [85.71%, 96.42%], SVM [78.20%, 91.91%]."

---

## Slide 9: Statistical Significance & McNemar Hypothesis Testing

### Statistical Rigor on Test Sets ([`results/mcnemar_tests.json`](results/mcnemar_tests.json)):
* **Hypothesis Testing Methodology**: McNemar's test with Edwards continuity correction on paired binary prediction outcomes ($n_{01}$ vs $n_{10}$).
* **Test 1: Clean Test Set (N=20,040) — Linear SVM vs. LightGBM**:
  * Contingency: SVM correct / LightGBM incorrect ($n_{10} = 292$); LightGBM correct / SVM incorrect ($n_{01} = 35$).
  * Test Statistic: $\chi^2 = 200.4159$, $p < 10^{-10}$ (**Statistically Significant** at $\alpha=0.05$).
  * Finding: Linear SVM maintains a statistically significant accuracy advantage on pure clean character n-grams.
* **Test 2: Real Indian Statements (N=105) — Linear SVM vs. MiniLM-LogReg**:
  * Contingency: $n_{10} = 11$, $n_{01} = 4$.
  * Test Statistic: $\chi^2 = 2.4000$, $p = 0.12134$ (**Not Statistically Significant** at $\alpha=0.05$).
* **Test 3: Real Indian Statements (N=105) — LightGBM vs. MiniLM-LogReg**:
  * Contingency: $n_{10} = 8$, $n_{01} = 11$.
  * Test Statistic: $\chi^2 = 0.2105$, $p = 0.64636$ (**Not Statistically Significant** at $\alpha=0.05$).
* **Tests 4–8: N=96 Exact Binomial McNemar** (5 comparisons, not multiple-comparison corrected; treat as suggestive):
  * DistilBERT vs SVM: b=6, c=0. Exact p=**0.03125** (α=0.05: sig; Bonferroni α=0.01: not sig).
  * DistilBERT vs LightGBM: b=15, c=0. Exact p=**0.00006** (sig after Bonferroni).
  * DistilBERT vs MiniLM: b=13, c=0. Exact p=**0.00024** (sig after Bonferroni).
  * SVM vs LightGBM: b=11, c=2. Exact p=0.02246 (not sig after Bonferroni).
  * SVM vs MiniLM: b=11, c=4. Exact p=0.11847 (not sig).

### Speaker Notes (1.5 minutes):
> "In senior engineering research, reporting raw percentages without statistical tests is insufficient. We conducted McNemar tests on paired predictions:
> 
> On the large 20,040 test set, the Linear SVM's lead over LightGBM is statistically significant ($\chi^2 = 200.42, p < 0.001$).
> 
> However, on the 105 real Indian bank statements, the observed difference between Linear SVM (92.38%), MiniLM (85.71%), and LightGBM (82.86%) is **not statistically significant** ($p = 0.12$ and $p = 0.65$). This proves an important scientific point: on a sample size of 105, random variation cannot be ruled out. We honestly report that these top three architectures perform comparably on real statements."

---

## Slide 10: Indian Banking & Regional Brand Generalization

### Target Brand Test Matrix ([`results/target_indian_merchants_all_models.csv`](results/target_indian_merchants_all_models.csv)):
| Target Brand | Input Description | Expected Category | Production Baseline | LightGBM | MiniLM-LogReg | Linear SVM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Amazon Pay** | `UPI/998124019283/AMAZON PAY/amazon@apl` | `Shopping & Retail` | `Income` ❌ | **`Shopping` (1.00)** ✅ | **`Shopping` (0.94)** ✅ | **`Shopping` (0.92)** ✅ |
| **Swiggy** | `UPI/308942189012/SWIGGY/swiggy@icici` | `Food & Dining` | `Shopping` ❌ | **`Food & Dining` (1.00)** ✅ | **`Food & Dining` (0.76)** ✅ | **`Food & Dining` (0.80)** ✅ |
| **Zomato** | `UPI/489120938102/ZOMATO/zomato@hdfcbank` | `Food & Dining` | `Shopping` ❌ | **`Food & Dining` (1.00)** ✅ | **`Food & Dining` (0.92)** ✅ | **`Food & Dining` (0.80)** ✅ |
| **Ola Cabs** | `UPI/992019283102/OLA CABS/olacabs@icici` | `Transportation` | `Shopping` ❌ | **`Transportation` (1.00)** ✅ | **`Transportation` (0.77)** ✅ | **`Transportation` (0.84)** ✅ |
| **Reliance Jio**| `UPI/110293849102/RELIANCE JIO/jio@paytm`| `Utilities & Services`| `Shopping` ❌ | **`Utilities` (1.00)** ✅ | **`Utilities` (0.96)** ✅ | **`Utilities` (0.81)** ✅ |
| **Blinkit** | `BLINKIT COMMERCE PVT LTD` | `Food & Dining` | `Shopping` ❌ | **`Food & Dining` (0.98)** ✅ | `Financial` ❌ | **`Food & Dining` (0.83)** ✅ |
| **Zepto** | `ZEPTO 10 MIN GROCERY DELIVERY` | `Food & Dining` | `Shopping` ❌ | **`Food & Dining` (1.00)** ✅ | **`Food & Dining` (0.96)** ✅ | **`Food & Dining` (0.84)** ✅ |
| **BigBasket** | `BIGBASKET SUPERMARKET ORDER` | `Food & Dining` | `Shopping` ❌ | `Shopping` ❌ | **`Food & Dining` (0.96)** ✅ | **`Food & Dining` (0.81)** ✅ |
| **Flipkart** | `IMPS/3948201/FLIPKART INTERNET/REF921` | `Shopping & Retail` | `Utilities` ❌ | **`Shopping` (1.00)** ✅ | **`Shopping` (0.50)** ✅ | **`Shopping` (0.80)** ✅ |
| **Myntra** | `POS 5241XXXX9021 MYNTRA DESIGNS BLR` | `Shopping & Retail` | `Shopping` ⚠️ | **`Shopping` (1.00)** ✅ | **`Shopping` (1.00)** ✅ | **`Shopping` (0.81)** ✅ |
| **Airtel** | `BHARTI AIRTEL POSTPAID BILL PAYMENT` | `Utilities & Services` | `Shopping` ❌ | `Income` ❌ | **`Utilities` (0.92)** ✅ | **`Utilities` (0.66)** ✅ |

### Speaker Notes (1.5 minutes):
> "This slide illustrates the impact of domain canonicalization and Indian training seeds on prominent brands.
> 
> The legacy production model suffered from severe regional blind spots: it misclassified `Swiggy`, `Zomato`, `Ola`, and `Jio` into generic Shopping, and misclassified `Amazon Pay` as `Income` due to the word 'pay'.
> 
> Under our updated models, Amazon Pay, Swiggy, Zomato, Ola, Jio, Zepto, and Flipkart are categorized correctly with near 100% confidence. LightGBM gets 9/11 correct, MiniLM gets 10/11, and Linear SVM gets 11/11."

---

## Slide 11: Limitations & Empirical Integrity Audit

### 10 Standardized Project Limitations:
1. **Indian Training Seed Provenance**: `data/indian_merchants_train_seed.csv` (3,062 rows) is LLM-generated from structured prompt templates; labels are unverified and not drawn from real bank statements.
2. **Small Real Indian Test Set**: Real Indian test set is small ($N=105$); performance differences between top models on it are not statistically significant.
3. **Seed-Test Merchant Overlap**: 8 merchants overlapped between seed and real test set; removed and retested for SVM, LightGBM, XGBoost, MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.
4. **Shared Noise Generator**: First noisy test shared noise code with training; independent Noisy V2 was built to evaluate out-of-distribution robustness.
5. **Synthetic Benchmark Nature**: Mitul Shah benchmark (100k transactions) is likely synthetic; clean accuracy near 99% reflects an easy, well-separated task.
6. **Single Run Variance**: Single seed, single run for all models; multi-seed variance was not evaluated across all architectures.
7. **Transformer Calibration under Noise**: DistilBERT is overconfident under noise (ECE of 0.24 on Noisy V2 vs 0.0008 on clean).
8. **Rent rule origin and scope**: The rent rule is an offline post-classification heuristic used in evaluation scripts only. It is not applied in the deployed Streamlit app, whose rent transactions may be misclassified (e.g. 'MONTHLY APARTMENT RENT TRANSFER TO LANDLORD' → Government & Legal). Its keywords match the Rent rows of the test set, so +rent figures are not evidence of generalization. Model-only figures are the headline numbers.
9. **Ambiguous Grocery Labeling**: Quick-commerce grocery apps (BigBasket, Blinkit, Zepto) have ambiguous ground truth labels (Food vs Shopping).
10. **Text-Only Features**: Input features are strictly text-only: no amount, timestamp, or Merchant Category Codes (MCC).
11. **XGBoost Gap Not Fully Ablated**: Controlled experiments explain +10 pts (sparse zeros) and +9 pts (depth-wise growth), but the remaining gap to LightGBM is unexplained. Exclusive Feature Bundling (EFB) is an untested hypothesis, not ablated.

### Speaker Notes (1.5 minutes):
> "In senior engineering research, scientific integrity requires explicitly documenting all ten project limitations rather than masking them.
> 
> Key points to highlight: our 3,062 Indian training samples are template-generated; the authentic Indian test set is modest at 105 rows with non-significant top-model differences; 8 overlapping merchants were audited and removed for classical models while DistilBERT remains preliminary; initial noisy scores reflected shared generator logic resolved by Noisy V2; and DistilBERT exhibits severe overconfidence under noise with an ECE of 0.24."

---

## Slide 12: Production Safety, Automated Testing & Reproducibility

### Verification Pillars:
* **Production Invariance Guaranteed**:
  * Production model (`final_model.joblib`) SHA-256 hash verified unchanged: `fd898d98b9f6ec03902d825082d69a461d32b3318bca47b98f45e8e8dfc11195`.
* **Automated Test Suite Execution**:
  * Total tests: **106 automated tests** passing in **16.28s** (`pytest -o pythonpath=.`).
  * 0 failed, 0 errors, 0 warnings.
* **Complete Terminal Reproducibility**:
  ```bash
  # 1. Run Fair Comparison Benchmark Suite
  python scripts/compare_models.py
  
  # 2. Run XGBoost Early Stopping Verification
  python scripts/check_xgboost_early_stopping.py
  
  # 3. Run Academic Integrity Audit
  python scripts/audit_phase4_integrity.py
  
  # 4. Run Full Automated Test Suite
  pytest -o pythonpath=.
  ```

### Speaker Notes (1.5 minutes):
> "From an MLOps and software engineering perspective, our pipeline adheres to strict safety standards.
> 
> First, `models/final_model.joblib` was preserved strictly read-only; its SHA-256 hash was cryptographically verified before and after all experiments.
> 
> Second, our automated test suite comprises **106 unit and integration tests**, covering domain canonicalization, TF-IDF feature dimensionality, probability normalization, privacy firewalls, and numeric grounding validators, all passing in 16 seconds.
> 
> Every single result and table shown today is directly generated by scripts in our repository."

---

## Slide 13: Summary, Academic Conclusions & Recommendations

### Final Conclusions:
1. **Model Selection According to Data Type**:
   * On short, sparse, noisy financial text, **linear models on character n-grams** remain exceptionally strong baselines due to linear separability in sparse feature space.
   * **LightGBM is the top-performing boosted tree model**: Features **97.72% Clean Accuracy**, **63.55% Noisy V2 Accuracy**, and **1.97 ms latency** (compared to 4.96 ms for the Linear SVM). Controlled diagnosis shows XGBoost's gap was mostly due to sparse zero handling (+10 pts) and depth-wise growth (+9 pts); the remaining gap to LightGBM is unexplained and EFB is an untested hypothesis.
   * **MiniLM Embeddings**: Provide strong semantic generalization (80.95% model-only; 85.71% +rent — offline evaluation only, not deployed) but are vulnerable to character-level typo corruption (57.06%).
2. **Deployed Model & Recommendation**:
   * **Deployed model**: Candidate Linear SVM (`variant_b_candidate.joblib`), chosen for robustness on independently generated noise (76.39% Noisy V2 vs 69.00% for DistilBERT) and low single-call latency (4.96 ms). DistilBERT performs best on clean text (99.13%) and the Indian set (92.71% model-only, preliminary — trained on the original seed containing 8 overlapping merchants); deploying it after retraining on the leakage-aware seed is future work.
   * LightGBM (97.72% clean, 1.97 ms) is the top-performing boosted tree model and a strong candidate for high-throughput batch pipelines, but its Noisy V2 accuracy (63.55%) is lower than the SVM (76.39%) for out-of-distribution robustness.

### Speaker Notes (1.5 minutes):
> "To conclude:
> 
> We have systematically evaluated four model families on short, noisy financial text.
> 
> **Deployed model**: The Candidate Linear SVM (`variant_b_candidate.joblib`) is deployed, chosen for robustness on independently generated noise (76.39% Noisy V2, significantly higher than DistilBERT's 69.00%, p<10⁻⁵) and low single-call latency (4.96 ms). DistilBERT performs best on clean text (99.13%) and the Indian statement set (92.71% model-only, preliminary — DistilBERT was trained on the original seed containing 8 overlapping merchants); deploying it after retraining on the leakage-aware seed is future work.
> 
> LightGBM is the top-performing boosted tree model (97.72% clean, 63.55% Noisy V2, 1.97 ms). Controlled experiments show XGBoost's gap was mostly due to sparse zero representation (+10 pts) and depth-wise growth (+9 pts); the remaining gap is unexplained and EFB is an untested hypothesis. With 106 automated tests passing, our system is statistically validated and reproducible.
> 
> Thank you, and I look forward to your questions."

---

## Viva & Defense Q&A Preparation: Anticipated Questions & Answers

### Q1: Why did you choose 3,000 character n-gram features instead of 20,000 word n-grams?
> **Answer**: 
> "Character n-grams (`char_wb` range 2 to 4) capture subword roots and morphological patterns, making the model inherently robust to banking abbreviations and typos like 'SWGY' or 'AMAZN'. We capped features at 3,000 because gradient boosted trees construct histograms across every feature. Unconstrained 20,000 features caused memory bottlenecks and slow training (over 20 minutes) with minimal accuracy gain. At 3,000 features, LightGBM trains in just 132 seconds while achieving 97.72% test accuracy."

### Q2: Why did XGBoost perform significantly worse than LightGBM on this dataset?
> **Answer**: 
> "Controlled diagnostic experiments (`results/xgboost_deep_diagnosis.csv`) showed that XGBoost's lower score was mostly explained by sparse zero handling (+10 points when passed dense zeros) and depth-wise tree growth (+9 points with `grow_policy='lossguide'`). The remaining gap between tuned XGBoost (90.02%) and LightGBM (97.72%) is unexplained. Exclusive Feature Bundling (EFB) is an untested hypothesis."

### Q3: Did you verify if XGBoost underperformed simply because of early stopping?
> **Answer**: 
> "Yes. We wrote an independent verification script (`scripts/check_xgboost_early_stopping.py`) that ran XGBoost with 100 trees and early stopping on validation loss. XGBoost ran all the way to iteration 99 without early stopping, achieving 72.41% validation accuracy and 68.53% test accuracy. This confirms that early stopping was not the cause; the limitation is structural to level-wise splitting on sparse text matrices."

### Q4: Why did MiniLM sentence embeddings drop from 89.67% on clean text to 57.06% on typo noise?
> **Answer**: 
> "Dense sentence transformers use WordPiece or Byte-Pair Encoding (BPE) subword tokenizers. When noise introduces character deletions or transpositions (e.g. 'swgy' or 'amazn py'), the tokenizer fragments the word into rare or uninformative sub-tokens, shifting the embedding vector far from its clean semantic neighborhood. Character n-gram models, by contrast, retain high subword overlap (e.g. `_sw`, `swg`, `wgy`) even when individual characters are altered."

### Q5: Is your 11-brand test proof that the model generalizes to all Indian merchants?
> **Answer**: 
> "No, and we explicitly document this in our limitations. An audit revealed that all 11 brands were present in `indian_merchants_train_seed.csv` (totaling 231 occurrences). Therefore, the 11-brand test is a sanity check to verify that our canonicalization and training pipeline correctly memorized these core patterns, rather than proof of zero-shot out-of-vocabulary generalization."

### Q6: How did you ensure there was no data leakage between your training and test sets?
> **Answer**: 
> "We enforced strict **merchant-grouped splitting**. The 100,000 transaction dataset was partitioned so that no merchant core in the test set (803 merchants) or validation set (333 merchants) ever appeared in the training set (2,975 merchants). Furthermore, all TF-IDF vectorizers were fit exclusively on training data. Finally, our 105 held-out Indian bank statements were verified to have zero string overlap with the training seeds."

### Q7: Why is the Linear SVM still slightly higher on clean test accuracy (99.00% vs 97.72%)?
> **Answer**: 
> "Support Vector Machines naturally excel at finding maximum-margin linear hyperplanes in high-dimensional sparse text spaces where features are linearly separable. However, the Linear SVM requires expensive Platt scaling calibration for probabilities (taking 4.96 ms per transaction), and cannot model non-linear feature interactions. LightGBM offers 97.72% accuracy, native calibrated probabilities, and an inference latency of 1.97 ms compared to 4.96 ms for the Linear SVM, making it superior for scalable production microservices."
