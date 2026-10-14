# Experimental Report: Transaction Canonicalization Evaluation

**Experiment Objective**:  
Evaluate the impact of transaction canonicalization on the Logistic Regression + TF-IDF baseline under strictly controlled conditions, preventing data leakage, measuring per-noise-type and per-class performance, analyzing harmful normalization on validation data, and determining production viability.

---

## 1. Executive Summary & Controlled Conditions

This experiment evaluates three controlled variants on the fixed merchant-grouped dataset split (`data/mitulshah_split.csv`):

1. **Variant A (Raw Text Baseline)**:
   - **Input**: Raw transaction descriptions.
   - **Features**: Character TF-IDF (2–5 n-grams, 20,000 features, sublinear TF, case-sensitive).
   - **Model**: Logistic Regression ($C=10$, `max_iter=300`, fixed seed 42).
2. **Variant B (Canonicalized Text)**:
   - **Input**: Canonicalized transaction descriptions via `src/canonicalize.py` (stripping UPI VPAs, POS merchant slips, NEFT prefixes, IMPS reference tokens, and masked card numbers).
   - **Features**: Identical Character TF-IDF (2–5 n-grams, 20,000 features) fitted exclusively on canonicalized training data.
   - **Model**: Identical Logistic Regression ($C=10$, `max_iter=300`, fixed seed 42).
3. **Variant C (Canonicalized + Engineered Features)**:
   - **Input**: Canonicalized transaction descriptions.
   - **Features**: Character TF-IDF (20,000 features) + Word TF-IDF (1–2 word n-grams, 5,000 features) + 7 Handcrafted morphological/banking keyword features (`length`, `digit_ratio`, `upper_ratio`, `has_upi`, `has_neft`, `has_pos`, `has_imps`) scaled via `MaxAbsScaler`. Total feature dimension: **25,007 features**.
   - **Model**: Identical Logistic Regression ($C=10$, `max_iter=300`, fixed seed 42).

### Top-Line Results Summary

| Metric | Variant A (Raw Baseline) | Variant B (Canonicalized) | Variant C (Canonicalized + Engineered) | Best Performing Variant |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Test Accuracy** | 98.88% | **98.99%** (+0.11%) | 98.79% (-0.09%) | **Variant B** |
| **Clean Test Macro-F1** | 0.9888 | **0.9899** (+0.0011) | 0.9879 (-0.0009) | **Variant B** |
| **Noisy Test Accuracy** | 80.02% | **88.72%** (**+8.69%**) | **95.37%** (**+15.35%**) | **Variant C** |
| **Noisy Test Macro-F1** | 0.8001 | **0.8870** (**+0.0868**) | **0.9538** (**+0.1537**) | **Variant C** |
| **Held-Out IMPS Accuracy** | 67.74% | **89.94%** (**+22.20%**) | **99.18%** (**+31.44%**) | **Variant C** |
| **Held-Out NEFT Accuracy** | 77.81% | **90.79%** (**+12.98%**) | **98.73%** (**+20.92%**) | **Variant C** |
| **Single-Item Latency** | **0.728 ± 0.010 ms** | 0.796 ± 0.064 ms (+0.068 ms) | 1.836 ± 0.045 ms (+1.108 ms) | **Variant A / B** |
| **Training Time** | 12.62 s | 10.55 s | **9.69 s** | **Variant C** |

---

## 2. Leakage Prevention & Experimental Protocol

1. **Vocabulary & Vectorizer Fitting**:
   - `TfidfVectorizer.fit_transform()` was strictly executed **only on training data** (69,885 records).
   - Validation and test datasets (both clean and noisy) were transformed using frozen vectorizer vocabularies via `.transform()`.
2. **Merchant Cluster Separation**:
   - Training: 2,975 unique merchants.
   - Validation: 333 unique merchants.
   - Test: 803 unique merchants.
   - Overlap between splits: **0 merchants**.
3. **Out-of-Distribution Data Separation**:
   - `data/own_test_indian.csv` ($N=105$) was completely excluded from training, vectorizer vocabulary generation, and feature engineering.
4. **Controlled Evaluation Protocol**:
   - Equivalent splits, label encodings, training noise augmentations (35%), and random seeds (42) were applied across all three variants.

---

## 3. Detailed Performance Breakdown

### 3.1 Performance by Synthetic Noise Type (Noisy Test Set, $N=20,040$)

Each test record was assigned a specific synthetic corruption type. NEFT and IMPS transformations were strictly held out from training noise.

| Noise Type | Training Status | Record Count | Variant A (Raw) Acc | Variant B (Canon) Acc | Variant C (Eng) Acc | $\Delta (B - A)$ Acc | $\Delta (C - A)$ Acc | Variant A F1 | Variant B F1 | Variant C F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPS** | **YES (Held-Out)** | 2,554 | 67.74% | **89.94%** | **99.18%** | **+22.20%** | **+31.44%** | 0.6809 | 0.8985 | 0.9917 |
| **NEFT** | **YES (Held-Out)** | 2,519 | 77.81% | **90.79%** | **98.73%** | **+12.98%** | **+20.92%** | 0.7787 | 0.9039 | 0.9873 |
| **UPI** | Seen in Training | 2,468 | 81.44% | **91.33%** | **98.82%** | **+9.89%** | **+17.38%** | 0.8103 | 0.9114 | 0.9878 |
| **POS** | Seen in Training | 2,504 | 82.27% | **91.05%** | **98.84%** | **+8.79%** | **+16.57%** | 0.8249 | 0.9102 | 0.9881 |
| **UPPERCASE** | Seen in Training | 2,460 | 82.56% | **90.61%** | **98.94%** | **+8.05%** | **+16.38%** | 0.8294 | 0.9072 | 0.9899 |
| **TRUNCATION**| Seen in Training | 2,487 | 69.92% | **77.52%** | **84.72%** | **+7.60%** | **+14.80%** | 0.7005 | 0.7778 | 0.8478 |
| **TYPO** | Seen in Training | 2,492 | 90.21% | **90.81%** | 85.75% | **+0.60%** | -4.45% | 0.9016 | 0.9074 | 0.8580 |
| **LOWERCASE** | Seen in Training | 2,556 | 88.38% | 87.68% | **97.85%** | -0.70% | **+9.47%** | 0.8848 | 0.8772 | 0.9790 |

**Observations**:
- Canonicalization alone (Variant B) delivers massive improvements on banking prefix wrappers, especially those never encountered during training (**IMPS: +22.20%**, **NEFT: +12.98%**, **UPI: +9.89%**).
- Variant C (Canonicalized + Word TF-IDF + Handcrafted features) further elevates noisy accuracy across 7 out of 8 noise types, achieving **>98.7% accuracy on IMPS, NEFT, UPI, and POS**.
- On character permutation (typo) noise, Variant B slightly improves (+0.60%), but Variant C experiences a slight drop (-4.45%), because word-level tokenizers fragment under internal typos while character n-grams remain robust.

---

### 3.2 Per-Class Precision, Recall, and F1 (Noisy Test Set)

| Category | Support | Raw (A) Prec | Raw (A) Rec | Raw (A) F1 | Canon (B) Prec | Canon (B) Rec | Canon (B) F1 | Eng (C) Prec | Eng (C) Rec | Eng (C) F1 | $\Delta (B - A)$ F1 | $\Delta (C - A)$ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Charity & Donations** | 2,068 | 0.7912 | 0.8540 | 0.8214 | 0.8733 | 0.9236 | 0.8978 | 0.9631 | 0.9729 | 0.9680 | **+0.0764** | **+0.1466** |
| **Entertainment & Rec** | 1,969 | 0.9168 | 0.6998 | 0.7938 | 0.9007 | 0.8933 | 0.8970 | 0.9819 | 0.9629 | 0.9723 | **+0.1032** | **+0.1785** |
| **Financial Services** | 2,012 | 0.9166 | 0.8360 | 0.8744 | 0.9444 | 0.9200 | 0.9320 | 0.9771 | 0.9737 | 0.9754 | **+0.0576** | **+0.1010** |
| **Food & Dining** | 1,996 | 0.7909 | 0.9454 | 0.8613 | 0.8773 | 0.9885 | 0.9296 | 0.9438 | 0.9925 | 0.9675 | **+0.0683** | **+0.1062** |
| **Government & Legal** | 2,009 | 0.6883 | 0.8387 | 0.7561 | 0.8085 | 0.8681 | 0.8373 | 0.9629 | 0.9428 | 0.9527 | **+0.0811** | **+0.1966** |
| **Healthcare & Medical**| 2,001 | 0.9215 | 0.7631 | 0.8349 | 0.9679 | 0.8751 | 0.9192 | 0.9696 | 0.8931 | 0.9298 | **+0.0843** | **+0.0949** |
| **Income** | 1,983 | 0.7757 | 0.9138 | 0.8391 | 0.9496 | 0.9415 | 0.9456 | 0.9744 | 0.9581 | 0.9662 | **+0.1065** | **+0.1271** |
| **Shopping & Retail** | 2,009 | 0.6311 | 0.5884 | 0.6090 | 0.7635 | 0.7088 | 0.7352 | 0.8909 | 0.9353 | 0.9126 | **+0.1262** | **+0.3036** |
| **Transportation** | 1,996 | 0.7231 | 0.7249 | 0.7240 | 0.8249 | 0.8191 | 0.8220 | 0.9061 | 0.9719 | 0.9379 | **+0.0980** | **+0.2139** |
| **Utilities & Services**| 1,997 | 0.9446 | 0.8368 | 0.8874 | 0.9754 | 0.9339 | 0.9542 | 0.9785 | 0.9339 | 0.9557 | **+0.0668** | **+0.0683** |

**Observations**:
- The largest per-class F1 gains from canonicalization occur in **Shopping & Retail (+12.62% in B, +30.36% in C)** and **Transportation (+9.80% in B, +21.39% in C)**, where Indian banking prefixes like `POS ... MUMBAI` and `UPI ... @okaxis` frequently caused confusion with Utilities or Financial Services.

---

### 3.3 Inference Latency Methodology & Measurements

**Measurement Protocol**:
- 200 warmup single-transaction inference iterations.
- 1,000 sequential single-item end-to-end predictions (preprocessing/canonicalization + feature transform + `predict_proba`).
- Repeated across 5 independent trials to calculate sample mean and standard deviation.

| Variant | Mean Latency (ms/txn) | Std Dev (ms) | Throughput (txns/sec) | Latency Overhead vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Variant A (Raw Text)** | **0.728 ms** | 0.010 ms | ~1,374 txns/s | Baseline reference |
| **Variant B (Canonicalized)** | **0.796 ms** | 0.064 ms | ~1,256 txns/s | **+0.068 ms (+9.3%)** |
| **Variant C (Canon + Engineered)** | **1.836 ms** | 0.045 ms | ~545 txns/s | **+1.108 ms (+152%)** |

**Conclusion on Latency**:
- Canonicalization alone introduces negligible latency overhead (**+0.068 ms** per transaction, running at >1,250 transactions/second on a single CPU core).
- Variant C doubles feature transforms (char TF-IDF + word TF-IDF + regex scanning for handcrafted flags), increasing latency to 1.836 ms. While still fast, Variant B offers superior computational efficiency.

---

## 4. Harmful Normalization Audit (Step 5, Validation Set Only)

To prevent test-set tuning and detect whether canonicalization removes useful discriminative tokens, we analyzed the held-out validation set ($N = 10,075$ records):

### Quantitative Validation Findings
- **Total validation records evaluated**: 10,075
- **Beneficial Normalizations** (Raw was incorrect $\rightarrow$ Canonicalized was correct): **261 records (2.59%)**
- **Harmful Normalizations** (Raw was correct $\rightarrow$ Canonicalized was incorrect): **57 records (0.57%)**
- **Net Normalized Advantage**: **+204 records (+2.02% net validation gain)**

### Qualitative Breakdown of Harmful Normalizations

| Validation Idx | True Category | Raw Noisy Description | Canonicalized Description | Raw Model Pred | Canon Model Pred | Root Cause Analysis |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **465** | Food & Dining | `UPI/415016293817/DINING OUT - SUSHI RESTAURANT/diningouts@okaxis` | `DINING OUT - SUSHI RESTAURANT` | Food & Dining *(Correct)* | Financial Services *(Incorrect)* | The VPA handle `@okaxis` / `diningouts` contained lower-case dining tokens. When stripped to uppercase `DINING OUT`, the token `OUT` elevated Financial Services prior. |
| **33** | Utilities & Services | `DSL #3418` | `DSL #3418` | Utilities & Services *(Correct)* | Government & Legal *(Incorrect)* | No text change occurred, but canonicalized model learned slightly different global n-gram weights for short 3-letter acronyms. |
| **83** | Shopping & Retail | `h&m #1655 store` | `h&m #1655 store` | Shopping & Retail *(Correct)* | Government & Legal *(Incorrect)* | Lowercase store name without banking noise; decision boundary shifted due to changed training distribution. |
| **568** | Utilities & Services | `DSL TXN270367` | `DSL TXN270367` | Utilities & Services *(Correct)* | Shopping & Retail *(Incorrect)* | Short acronym with transaction ID; without noise wrappers, `TXN` token favored retail classification. |

**Safeguard Implemented in Production**:  
`src/canonicalize.py` includes a fallback safeguard: if regex stripping collapses a description into an empty string or eliminates all alphabetical characters, the preprocessor automatically reverts to the original trimmed text.

---

## 5. Architectural Recommendation & Production Readiness

### Comparison Summary
1. **Variant A (Raw Baseline)**: Fails under domain noise (80.02% noisy test accuracy; down to 67.74% on IMPS).
2. **Variant B (Canonicalized)**: Elevates noisy test accuracy from **80.02% to 88.72% (+8.69%)**, recovers held-out IMPS by **+22.20%** and held-out NEFT by **+12.98%**, maintains 98.99% clean accuracy, and adds negligible latency (**+0.068 ms**).
3. **Variant C (Canonicalized + Engineered Features)**: Delivers highest noisy accuracy (**95.37%**), but increases latency by **+1.108 ms** and suffers a slight drop on character-typo noise (85.75% vs 90.81%).

### Final Recommendation
**The measured improvement strongly justifies integrating transaction canonicalization (Variant B) into the production pipeline:**
- It is computationally lightweight (<0.07 ms overhead).
- It introduces zero breaking changes to existing model contracts.
- It provides dramatic out-of-distribution robustness on unseen Indian banking wrappers (+22.2% on IMPS, +13.0% on NEFT).
- For environments where maximum noise recovery is prioritized over sub-millisecond throughput, **Variant C** provides a compelling upgrade path to 95.37% noisy test accuracy.
