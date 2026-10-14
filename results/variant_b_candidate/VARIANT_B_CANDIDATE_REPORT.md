# Experimental Variant B Candidate Evaluation Report (Phase 2: Indian Merchant Seed Retraining)

**Author**: Senior Machine Learning Engineer  
**Date**: October 12, 2026  
**Artifact Evaluated**: [`models/variant_b_candidate.joblib`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/models/variant_b_candidate.joblib) (SHA256: `bdb1aa42becbae67f98b428d22fd61303f3ec14cd7cb90aaa49686ef6acc3e8a`)  
**Baseline Compared**: Production Artifact [`models/final_model.joblib`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/models/final_model.joblib) (SHA256: `fd898d98b9f6ec03902d825082d69a461d32b3318bca47b98f45e8e8dfc11195`)  
**Results Directory**: [`results/variant_b_candidate/`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/results/variant_b_candidate/)  

---

## 1. Executive Summary & Staging Recommendation

### Final Verdict: **SAFE TO PROCEED TO STAGING**

The retrained **Variant B Candidate Model** with the leakage-safe Indian merchant training seed (`data/indian_merchants_train_seed.csv`, 3,062 rows) and unified preprocessing (`TextPreprocessor(use_canonicalization=True)` + `TfidfVectorizer(lowercase=True)`) has solved all previously identified failure modes while maintaining clean test parity on unseen merchants.

### Key Milestones Achieved:

1. **Target Indian Merchant Brand Resolution (11/11 Correct - 100.0%)**:
   * **Amazon Pay**: `Shopping & Retail` (0.9184) vs Production `Income` (0.5000) [CRITICAL REGRESSION RESOLVED]
   * **Swiggy**: `Food & Dining` (0.8027) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Zomato**: `Food & Dining` (0.8015) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Ola**: `Transportation` (0.8388) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Jio**: `Utilities & Services` (0.8148) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Blinkit**: `Food & Dining` (0.8311) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Zepto**: `Food & Dining` (0.8370) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **BigBasket**: `Food & Dining` (0.8074) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Flipkart**: `Shopping & Retail` (0.7995) vs Production `Utilities & Services` (0.9994) [RESOLVED]
   * **Myntra**: `Shopping & Retail` (0.8085) vs Production `Shopping & Retail` (0.6385) [RESOLVED]
   * **Airtel**: `Utilities & Services` (0.6645) vs Production `Shopping & Retail` (0.6385) [RESOLVED]

2. **Real Indian Banking Statements ([`data/own_test_indian.csv`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/data/own_test_indian.csv), N=105)**:
   * **Model-Only Accuracy**: Jumped from **56.19%** (Production) to **87.62%** (Candidate) — **+31.43% absolute improvement**.
   * **Model + Rent Rule Accuracy**: Reached **92.38%** vs Production 60.95% (**+31.43% absolute improvement**).
   * **Real Statement UPI Accuracy**: **100.0% (6/6)** vs Production 16.67% (1/6) (**+83.33% improvement**).
   * **Real Statement IMPS Accuracy**: **100.0% (2/2)** vs Production 50.0% (1/2).

3. **Untouched Unseen-Merchant Test Benchmark ([`data/mitulshah_split.csv`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/data/mitulshah_split.csv), N=20,040)**:
   * **Clean Test Accuracy**: **99.00%** (Macro-F1 0.9900) vs Production 99.07% (Macro-F1 0.9907) — clean accuracy parity preserved (-0.07% negligible delta).
   * **Noisy Test Accuracy**: **97.19%** (Macro-F1 0.9719) vs Production 92.73% (Macro-F1 0.9326) — **+4.46% absolute improvement across 20,040 noisy transactions**.
   * **Typo Noise Accuracy**: **89.53%** vs Production 68.10% (**+21.43% absolute improvement**).
   * **Truncation Noise Accuracy**: **93.53%** vs Production 78.73% (**+14.80% absolute improvement**).

4. **Production Safety & Efficiency**:
   * **Production Model Artifact Invariance**: SHA256 `FD898D98B9F6EC03902D825082D69A461D32B3318BCA47B98F45E8E8DFC11195` verified byte-identical before and after evaluation.
   * **Artifact Footprint**: Reduced by 22.5% (**8.29 MB** vs 10.69 MB).
   * **Inference Latency**: **4.50 ms/txn** (faster than production's 4.69 ms/txn).
   * **Test Suite**: **93/93 tests passing**.

---

## 2. Dataset Provenance & Leakage Audit

### Audit Findings

1. **Merchant-Group Split Integrity (`data/mitulshah_split.csv`)**:
   * Base Train: 69,885 rows (2,975 merchants)
   * Base Val: 10,075 rows (333 merchants)
   * Base Test: 20,040 rows (803 merchants)
   * Cross-split merchant overlap: **Strictly 0**.
2. **Indian Merchant Seed Provenance (`data/indian_merchants_train_seed.csv`)**:
   * Sample Count: 3,062 synthetic Indian merchant transactions.
   * Coverage: Balanced across all 10 target categories (~300 samples/category).
   * **Leakage Assertion Check**:
     $$\text{Overlap}\left(\text{train\_seed},\; \text{own\_test\_indian.csv}\right) = \emptyset$$
     Verified at runtime: **0 overlapping descriptions** with the 105 held-out evaluation transactions.
   * Augmented Training Split: $69,885 + 3,062 = \mathbf{72,947}$ records.
3. **Synthetic Noise Isolation**:
   * Train split augmentation (35% rate) utilized deterministic PRNG seed 42.
   * Validation split augmentation utilized PRNG seed 11.
   * Test split augmentation utilized PRNG seed 7.
   * Zero noise-pattern leakage across splits.

---

## 3. Architecture & Hyperparameters

* **Preprocessing Transformation**:
  ```python
  TextPreprocessor(
      lowercase=True,
      normalize_whitespace=True,
      use_canonicalization=True
  ).clean_text(desc)
  ```
* **Feature Extraction**:
  ```python
  TfidfVectorizer(
      analyzer="char_wb",
      ngram_range=(2, 5),
      min_df=3,
      max_features=20000,
      sublinear_tf=True,
      lowercase=True
  )
  ```
  * Vocabulary Size: 20,000 subword character n-grams.
  * Fit Time: 2.38s.
* **Classifier & Calibration**:
  ```python
  base_svc = LinearSVC(C=10.0, max_iter=2000, random_state=42)
  candidate_model = CalibratedClassifierCV(base_svc, method="sigmoid")
  ```
  * Fit Time: 46.64s.
  * Validation Accuracy: **0.9892** | Validation Macro-F1: **0.9892**.

---

## 4. Fair Benchmark Comparison

Both models evaluated under identical prediction conditions via [`scripts/train_and_evaluate_variant_b_indian.py`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/scripts/train_and_evaluate_variant_b_indian.py):

### Benchmark Summary Table ([`production_vs_candidate_benchmark.csv`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/results/variant_b_candidate/production_vs_candidate_benchmark.csv))

| Metric | Production Baseline (`final_model.joblib`) | Candidate Variant B (`variant_b_candidate.joblib`) | Delta |
| :--- | :---: | :---: | :---: |
| **mitulshah Clean Test Accuracy** | **0.9907** | 0.9900 | -0.0006 |
| **mitulshah Clean Test Macro-F1** | **0.9907** | 0.9900 | -0.0007 |
| **mitulshah Noisy Test Accuracy** | 0.9273 | **0.9719** | **+0.0446** |
| **mitulshah Noisy Test Macro-F1** | 0.9326 | **0.9719** | **+0.0393** |
| **mitulshah Typo Accuracy** | 0.6810 | **0.8953** | **+0.2143** |
| **mitulshah Truncation Accuracy** | 0.7873 | **0.9353** | **+0.1480** |
| **own_test_indian Model-Only Acc** | 0.5619 | **0.8762** | **+0.3143** |
| **own_test_indian Model+RentRule Acc** | 0.6095 | **0.9238** | **+0.3143** |
| **Inference Latency (ms/txn)** | 4.692 | **4.496** | **-0.196** |
| **Model Artifact Size (MB)** | 10.69 MB | **8.29 MB** | **-2.40 MB** |

### Breakdown by Synthetic Noise Condition ([`mitulshah_noise_type_breakdown.csv`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/results/variant_b_candidate/mitulshah_noise_type_breakdown.csv))

| Noise Type | Test Count | Production Accuracy | Candidate Accuracy | Candidate Macro-F1 | Accuracy Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Typo** | 2,492 | 0.6810 | **0.8953** | **0.8942** | **+0.2143** |
| **Truncation** | 2,487 | 0.7873 | **0.9353** | **0.9363** | **+0.1480** |
| **Lowercase** | 2,556 | 0.9894 | **0.9918** | **0.9920** | **+0.0023** |
| **Uppercase** | 2,460 | 0.9915 | **0.9915** | **0.9919** | 0.0000 |
| **UPI** | 2,468 | 0.9923 | 0.9907 | 0.9903 | -0.0016 |
| **POS** | 2,504 | 0.9912 | 0.9908 | 0.9905 | -0.0004 |
| **NEFT** | 2,519 | 0.9917 | 0.9889 | 0.9889 | -0.0028 |
| **IMPS** | 2,554 | 0.9910 | 0.9902 | 0.9902 | -0.0008 |

### Real Indian Banking Subsets ([`own_test_subsets_comparison.csv`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/results/variant_b_candidate/own_test_subsets_comparison.csv))

| Noise Subset | Test Count | Production Accuracy | Candidate Accuracy | Accuracy Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Indian** | 93 | 0.6237 | **0.9140** | **+0.2903** |
| **Real UPI** | 6 | 0.1667 | **1.0000** | **+0.8333** |
| **Real IMPS** | 2 | 0.5000 | **1.0000** | **+0.5000** |
| **Real NEFT** | 2 | 1.0000 | **1.0000** | 0.0000 |
| **Real POS** | 2 | 1.0000 | **1.0000** | 0.0000 |

---

## 5. Target Indian Merchant Brand Verification

Complete test results from [`results/variant_b_candidate/indian_merchant_evaluation.csv`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/results/variant_b_candidate/indian_merchant_evaluation.csv):

| Merchant | Raw Input Transaction | Expected Category | Production Prediction (Conf) | Candidate Prediction (Conf) | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Amazon Pay** | `UPI/998124019283/AMAZON PAY/amazon@apl` | `Shopping & Retail` | `Income` (0.5000) ❌ | **`Shopping & Retail` (0.9184)** ✅ | **PASSED** |
| **Swiggy** | `UPI/308942189012/SWIGGY/swiggy@icici` | `Food & Dining` | `Shopping & Retail` (0.6385) ❌ | **`Food & Dining` (0.8027)** ✅ | **PASSED** |
| **Zomato** | `UPI/489120938102/ZOMATO/zomato@hdfcbank` | `Food & Dining` | `Shopping & Retail` (0.6385) ❌ | **`Food & Dining` (0.8015)** ✅ | **PASSED** |
| **Ola** | `UPI/992019283102/OLA CABS/olacabs@icici` | `Transportation` | `Shopping & Retail` (0.6385) ❌ | **`Transportation` (0.8388)** ✅ | **PASSED** |
| **Jio** | `UPI/110293849102/RELIANCE JIO RECHARGE/jio@paytm` | `Utilities & Services` | `Shopping & Retail` (0.6385) ❌ | **`Utilities & Services` (0.8148)** ✅ | **PASSED** |
| **Blinkit** | `BLINKIT COMMERCE PVT LTD` | `Food & Dining` | `Shopping & Retail` (0.6385) ❌ | **`Food & Dining` (0.8311)** ✅ | **PASSED** |
| **Zepto** | `ZEPTO 10 MIN GROCERY DELIVERY` | `Food & Dining` | `Shopping & Retail` (0.6385) ❌ | **`Food & Dining` (0.8370)** ✅ | **PASSED** |
| **BigBasket** | `BIGBASKET SUPERMARKET ORDER` | `Food & Dining` | `Shopping & Retail` (0.6385) ❌ | **`Food & Dining` (0.8074)** ✅ | **PASSED** |
| **Flipkart** | `IMPS/3948201/FLIPKART INTERNET/REF921` | `Shopping & Retail` | `Utilities & Services` (0.9994) ❌ | **`Shopping & Retail` (0.7995)** ✅ | **PASSED** |
| **Myntra** | `POS 5241XXXX9021 MYNTRA DESIGNS BANGALORE` | `Shopping & Retail` | `Shopping & Retail` (0.6385) ⚠️ | **`Shopping & Retail` (0.8085)** ✅ | **PASSED** |
| **Airtel** | `BHARTI AIRTEL POSTPAID BILL PAYMENT` | `Utilities & Services` | `Shopping & Retail` (0.6385) ❌ | **`Utilities & Services` (0.6645)** ✅ | **PASSED** |

---

## 6. Artifact Integrity & Safety Verification

| Artifact Path | Expected Role | SHA256 Hash | Status |
| :--- | :--- | :--- | :---: |
| [`models/final_model.joblib`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/models/final_model.joblib) | Production Model | `fd898d98b9f6ec03902d825082d69a461d32b3318bca47b98f45e8e8dfc11195` | **Unmodified** |
| [`models/final_model_pre_canonicalization_backup.joblib`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/models/final_model_pre_canonicalization_backup.joblib) | Immutable Backup | `fd898d98b9f6ec03902d825082d69a461d32b3318bca47b98f45e8e8dfc11195` | **Unmodified** |
| [`models/variant_b_candidate.joblib`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/models/variant_b_candidate.joblib) | New Candidate Model | `bdb1aa42becbae67f98b428d22fd61303f3ec14cd7cb90aaa49686ef6acc3e8a` | **Verified** |

---

## 7. Next Steps & Staging Deployment Protocol

1. **Staging Promotion**:
   * Deploy [`models/variant_b_candidate.joblib`](file:///c:/Users/deeph/OneDrive/Desktop/AI-Personal-Finance-Advisor/models/variant_b_candidate.joblib) to the staging environment under feature flag `MODEL_VERSION=3.1.0-variant_b_indian_seed`.
2. **Shadow Traffic Evaluation**:
   * Route 10% of production traffic through Candidate Variant B in shadow mode. Log confidence and abstention distributions.
3. **Rollback Protocol**:
   * If any anomaly occurs in staging, revert `MODEL_VERSION` to production `2.0.0-calibrated_svm` pointing to `models/final_model.joblib`.
