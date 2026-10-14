import sys
from pathlib import Path

readme_path = Path("README.md")
lines = readme_path.read_text(encoding="utf-8").splitlines(keepends=True)

# Find start of section 3.1
idx_sec3 = -1
for i, line in enumerate(lines):
    if "### 3.1 4-Model Family Comparison Matrix" in line:
        idx_sec3 = i
        break

if idx_sec3 == -1:
    print("Could not find section 3.1")
    sys.exit(1)

# Find start of section 3.2
idx_sec32 = -1
for i, line in enumerate(lines):
    if "### 3.2 Canonicalization Gain Per Noise Type" in line:
        idx_sec32 = i
        break

if idx_sec32 == -1:
    print("Could not find section 3.2")
    sys.exit(1)

new_sec3_lines = [
    "### 3.1 4-Model Family Comparison Matrix\n",
    "\n",
    "| Model Architecture | Model Family | Status | Clean Test Acc | 95% Wilson CI | Noisy V2 Acc (Headline) | Typo Acc | Indian Model-Only (N=105 / 96 / 58) | Latency (ms) | Fit Time | Artifact Size |\n",
    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
    "| **Production Baseline** (`final_model.joblib`) | Linear (Legacy) | **MEASURED** | 0.9907 | `[98.92%, 99.19%]` | 0.6389 *(init: 0.9273)* | 0.6810 | 0.5619 / — / — | 4.76 ms | *Legacy* | 10.69 MB |\n",
    "| **Candidate Linear SVM** (`variant_b_candidate.joblib`) | Linear (SVM) | **MEASURED** | **0.9900** | `[98.85%, 99.13%]` | **0.7639** *(init: 0.9719)* | **0.8953** | **0.8762 / 0.8646 / 0.8103** | 4.96 ms | 46.6s | 8.29 MB |\n",
    "| **LightGBM Classifier** (`lightgbm_...joblib`) | Boosted Trees (GBDT) | **MEASURED** | **0.9772** | `[97.50%, 97.92%]` | **0.6355** *(init: 0.9309)* | **0.8054** | **0.7810 / 0.8021 / 0.7414** | **1.97 ms** | 132.9s | 6.96 MB |\n",
    "| **XGBoost Classifier** (`xgboost_...joblib`) | Boosted Trees (GBDT) | **MEASURED** | 0.7261 | `[71.99%, 73.23%]` | 0.4514 *(init: 0.6689)* | 0.5373 | 0.7048 / — / — | **1.55 ms** | 265.1s | 1.47 MB |\n",
    "| **MiniLM-L6-v2 + LogReg** (`minilm_...joblib`) | Dense Embeddings | **MEASURED** | 0.8967 | `[89.24%, 90.08%]` | 0.4781 *(init: 0.8282)* | 0.5706 | 0.8095 / — / — | 9.73 ms | 12.0s | 0.02 MB |\n",
    "| **DistilBERT Fine-Tuned** (`notebooks/05_...`) | Transformer | **MEASURED** | **0.9913** | `[98.99%, 99.25%]` | **0.6900** | 0.7752 | **0.9333 / 0.9271 / 0.8793** | (GPU batch) | 348.5s | 255.4 MB |\n",
    "\n",
    "*Notes on evaluation columns*:\n",
    "- **Noisy Test**: Headline noisy accuracy is evaluated on **Noisy V2** (independent noise generator, $N=20,040$). Initial noisy test scores *(in parentheses)* shared noise generator logic with training and are described only as initial, optimistic.\n",
    "- **Indian Statement Acc**: Primary reported metric is **Model-Only** accuracy across $N=105$ (full real test), $N=96$ (non-overlapping), and $N=58$ (fully disjoint). Rule-assisted (+Rent) accuracies are secondary: Linear SVM (+Rent): 92.38% ($N=105$), 91.67% ($N=96$), 89.66% ($N=58$); DistilBERT (+Rent): 98.10% ($N=105$), 97.92% ($N=96$), 96.55% ($N=58$); LightGBM (+Rent): 82.86% ($N=105$), 85.42% ($N=96$), 82.76% ($N=58$).\n",
    "- **Statistical Stance**: DistilBERT vs Linear SVM on Indian bank statements was **NOT statistically tested**. With overlapping confidence intervals on small samples, we do not claim DistilBERT beats the SVM on Indian data.\n",
    "\n",
    "*Architectural Findings by Data Type*:\n",
    "- **Linear Models (SVM)**: High-dimensional character n-grams are naturally linearly separable. The linear SVM achieves the highest typo robustness (89.53%) and clean accuracy (99.00%), outperforming complex models under independent character noise (76.39% on Noisy V2).\n",
    "- **Boosted Trees (LightGBM vs XGBoost)**: LightGBM is the top-performing boosted tree model (+25.11% clean accuracy over XGBoost; 1.97 ms vs 4.96 ms Linear SVM latency). Controlled experiments reveal that XGBoost's deficit was mostly due to sparse zero handling (+10 points when passed dense zeros) and depth-wise growth (+9 points with `grow_policy='lossguide'`); the remaining gap to LightGBM is unexplained. Exclusive Feature Bundling (EFB) remains an untested hypothesis.\n",
    "- **Dense Embeddings (MiniLM)**: Captures semantic nuance on real Indian bank statements (80.95% model-only / 85.71% +rent), but drops on typo noise (57.06%) because character substitutions break subword token representations.\n",
    "- **Fine-Tuned Transformers (DistilBERT)**: Contextual self-attention achieves peak clean test accuracy (99.13%) and strong Indian statement performance (93.33% model-only / 98.10% +rent), but drops to 69.00% under independent character noise with an ECE of 0.24, showing subword tokenizer fragility on character deletions. Fine-tuned on Kaggle in 348.5 s (effective batch 64 via 2x T4 GPUs, no gradient accumulation, lr 5e-5, 3 epochs, single seed 42).\n",
    "\n",
]

# Splice section 3.1
lines = lines[:idx_sec3] + new_sec3_lines + lines[idx_sec32:]

# Now find Section 4
idx_sec4 = -1
for i, line in enumerate(lines):
    if "## 4. Statistical Honesty & Integrity Audit" in line:
        idx_sec4 = i
        break

if idx_sec4 == -1:
    print("Could not find section 4")
    sys.exit(1)

# Find Section 5
idx_sec5 = -1
for i, line in enumerate(lines):
    if "## 5. Getting Started & Reproducibility" in line:
        idx_sec5 = i
        break

if idx_sec5 == -1:
    print("Could not find section 5")
    sys.exit(1)

new_sec4_and_limitations = [
    "## 4. Statistical Honesty & Integrity Audit\n",
    "\n",
    "1. **Real Indian Test Set ($N=105$, `data/own_test_indian.csv`)**:\n",
    "   - Primary Model-Only accuracies: DistilBERT: **93.33%** ($N=105$) / **92.71%** ($N=96$) / **87.93%** ($N=58$); Candidate Linear SVM: **87.62%** ($N=105$) / **86.46%** ($N=96$) / **81.03%** ($N=58$); MiniLM-LogReg: **80.95%** ($N=105$); LightGBM: **78.10%** ($N=105$) / **80.21%** ($N=96$) / **74.14%** ($N=58$); XGBoost: **70.48%** ($N=105$). Rule-assisted (+Rent) results are secondary.\n",
    "   - McNemar statistical tests reveal that differences between top models on the $N=105$ sample have $p$-values between 0.12 and 0.65 ($p > 0.05$). Differences between top models on this sample size are not statistically significant. DistilBERT vs Linear SVM on Indian bank statements was **NOT statistically tested**.\n",
    "2. **Synthetic Training Seed Provenance & Leakage Removal**:\n",
    "   - `data/indian_merchants_train_seed.csv` (3,062 rows) was generated using templates of known Indian merchants.\n",
    "   - 8 merchants overlapped between the training seed and the real test set; these were removed and retested for SVM, LightGBM, XGBoost, and MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.\n",
    "3. **Synthetic Noise Function Sharing**:\n",
    "   - The initial noisy test set shared noise code with training and is described only as \"initial, optimistic\" (e.g. Linear SVM 97.19%).\n",
    "   - An independent noise generator was built for **Noisy V2** ($N=20,040$), which serves as the headline noisy benchmark (Linear SVM: 76.39%, DistilBERT: 69.00%, LightGBM: 63.55%).\n",
    "\n",
    "---\n",
    "\n",
    "## 5. Limitations\n",
    "\n",
    "1. **Indian training seed provenance**: `data/indian_merchants_train_seed.csv` (3,062 rows) is generated from structured prompt templates, and labels are unverified.\n",
    "2. **Small real Indian test set**: The real Indian test set is small ($N=105$); performance differences between top models on it are not statistically significant.\n",
    "3. **Seed-test merchant overlap**: 8 merchants overlapped between the training seed and the real test set; these were removed and retested for SVM, LightGBM, XGBoost, and MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.\n",
    "4. **Shared noise generator**: The first noisy test set shared noise code with training; an independent noisy test set (Noisy V2) was built to evaluate out-of-distribution robustness.\n",
    "5. **Synthetic benchmark nature**: The Mitul Shah benchmark (100k transactions) is likely synthetic; clean accuracy near 99% reflects an easy, well-separated task.\n",
    "6. **Single run variance**: Benchmarks use a single seed and a single run for all models; multi-seed variance was not evaluated across all architectures.\n",
    "7. **Transformer calibration under noise**: DistilBERT is severely overconfident under character noise (Expected Calibration Error of 0.24 on Noisy V2).\n",
    "8. **Rent rule observation risk**: The heuristic rent rule may have been written after inspecting patterns in the 105 real statements.\n",
    "9. **Ambiguous grocery labeling**: Quick-commerce grocery apps (BigBasket, Blinkit, Zepto) exhibit ambiguous ground truth labels across Food & Dining vs Shopping & Retail.\n",
    "10. **Text-only inputs**: The classification engine is text-only and does not incorporate transaction amount, timestamp, or Merchant Category Codes (MCC).\n",
    "\n",
    "---\n",
    "\n",
    "## 6. Getting Started & Reproducibility\n",
]

lines = lines[:idx_sec4] + new_sec4_and_limitations + lines[idx_sec5 + 1:]

# Update other references
for i, line in enumerate(lines):
    if "## 6. Repository Structure" in line:
        lines[i] = "## 7. Repository Structure\n"
    elif "## 7. License" in line:
        lines[i] = "## 8. License\n"
    elif "(74 tests)" in line:
        lines[i] = line.replace("(74 tests)", "(106 tests)")

readme_path.write_text("".join(lines), encoding="utf-8")
print("README.md successfully updated!")
