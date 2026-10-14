"""
Integrity and Honesty Audit Script (Phase 4 + Security/PII)
Checks:
1. Merchant-level overlap between indian_merchants_train_seed.csv and own_test_indian.csv.
2. 11 Target brands frequency in the seed.
3. Noise generation traceability and function overlap.
4. Mismatch analysis between README.md and MODEL_EVALUATION_REPORT.md.
5. Label encoding and class ordering between LightGBM and XGBoost.
6. PII audit on own_test_indian.csv and secrets scan.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import joblib
import pandas as pd
from collections import Counter
from src.canonicalize import canonicalize
from src.preprocessing import TextPreprocessor

def run_merchant_overlap_check():
    print("=" * 60)
    print("1. MERCHANT-LEVEL OVERLAP CHECK")
    print("=" * 60)
    
    seed_path = "data/indian_merchants_train_seed.csv"
    test_path = "data/own_test_indian.csv"
    
    if not os.path.exists(seed_path) or not os.path.exists(test_path):
        print(f"Missing files: seed={os.path.exists(seed_path)}, test={os.path.exists(test_path)}")
        return
        
    df_seed = pd.read_csv(seed_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Seed rows: {len(df_seed)}, Test rows: {len(df_test)}")
    
    # Raw descriptions
    raw_seed_desc = set(df_seed["transaction_description"].str.strip().str.lower())
    raw_test_desc = set(df_test["transaction_description"].str.strip().str.lower())
    raw_overlap = raw_seed_desc.intersection(raw_test_desc)
    print(f"Exact raw description overlap: {len(raw_overlap)}")
    if raw_overlap:
        print(f"  Overlapping raw descriptions: {raw_overlap}")
    
    # Canonicalized descriptions
    preprocessor = TextPreprocessor(use_canonicalization=True)
    clean_seed_canonical = df_seed["transaction_description"].apply(lambda x: preprocessor.clean_text(str(x))).str.strip().tolist()
    clean_test_canonical = df_test["transaction_description"].apply(lambda x: preprocessor.clean_text(str(x))).str.strip().tolist()
    
    set_seed_canonical = set(clean_seed_canonical)
    set_test_canonical = set(clean_test_canonical)
    
    canonical_overlap = set_seed_canonical.intersection(set_test_canonical)
    print(f"Canonicalized text exact overlap: {len(canonical_overlap)} unique strings")
    if canonical_overlap:
        print("Overlapping canonicalized strings:")
        for s in sorted(canonical_overlap):
            # Show matching rows
            test_matches = df_test[df_test["transaction_description"].apply(lambda x: preprocessor.clean_text(str(x))).str.strip() == s]["transaction_description"].tolist()
            seed_count = sum(1 for x in clean_seed_canonical if x == s)
            print(f"  - '{s}': in seed {seed_count} times; in test as: {test_matches}")
            
    # Check 11 target brands in seed
    target_brands = [
        "swiggy", "zomato", "ola", "jio", "blinkit", "zepto", 
        "bigbasket", "flipkart", "myntra", "airtel", "amazon"
    ]
    print("\nTarget Brand Occurrence in Indian Merchant Training Seed (N=3,062):")
    seed_text_lower = df_seed["transaction_description"].str.lower()
    brand_counts = {}
    for brand in target_brands:
        count = seed_text_lower.str.contains(rf"\b{brand}\b", regex=True).sum()
        brand_counts[brand] = int(count)
        print(f"  - {brand:12s}: {count:4d} occurrences")
        
    total_brand_samples = sum(brand_counts.values())
    print(f"Total target brand occurrences in seed: {total_brand_samples}")

def run_noise_traceability():
    print("\n" + "=" * 60)
    print("2. NOISE GENERATION TRACEABILITY & FUNCTION OVERLAP")
    print("=" * 60)
    
    # Search for noise generation scripts
    print("Tracing noise generation in codebase...")
    scripts_to_check = [
        "scripts/run_canonicalization_experiment.py",
        "scripts/run_phase2_noise_robustness.py",
        "scripts/investigate_typo_noise.py",
        "scripts/compare_models.py",
        "src/noise.py",
        "src/noise_generator.py"
    ]
    for s in scripts_to_check:
        if os.path.exists(s):
            print(f"Found: {s}")
            with open(s, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Find function definitions related to noise
                func_defs = re.findall(r"def\s+([a-zA-Z0-9_]*noise[a-zA-Z0-9_]*|[a-zA-Z0-9_]*typo[a-zA-Z0-9_]*)\s*\(", content, re.IGNORECASE)
                seed_matches = re.findall(r"(?:seed|random_state)\s*=\s*(\d+)", content)
                print(f"  Functions: {set(func_defs)}")
                print(f"  Seeds found: {set(seed_matches[:5])}")

def run_xgboost_alignment_check():
    print("\n" + "=" * 60)
    print("3. XGBOOST VS LIGHTGBM LABEL ENCODING & ALIGNMENT")
    print("=" * 60)
    
    lgb_path = "models/lightgbm_transaction_classifier.joblib"
    xgb_path = "models/xgboost_transaction_classifier.joblib"
    
    if os.path.exists(lgb_path) and os.path.exists(xgb_path):
        lgb_art = joblib.load(lgb_path)
        xgb_art = joblib.load(xgb_path)
        
        lgb_classes = list(lgb_art.get("classes_", []))
        xgb_classes = list(xgb_art.get("classes_", []))
        
        print(f"LightGBM classes: {lgb_classes}")
        print(f"XGBoost classes:  {xgb_classes}")
        print(f"Classes identical: {lgb_classes == xgb_classes}")
        
        # Check vectorizer
        lgb_vocab = lgb_art["vectorizer"].vocabulary_
        xgb_vocab = xgb_art["vectorizer"].vocabulary_
        print(f"LGB vocab size: {len(lgb_vocab)}, XGB vocab size: {len(xgb_vocab)}")
        print(f"Vocabularies identical: {lgb_vocab == xgb_vocab}")
    else:
        print("One or both model artifacts missing.")

def run_security_pii_check():
    print("\n" + "=" * 60)
    print("4. SECURITY, SECRETS, AND PII CHECK")
    print("=" * 60)
    
    # Scan own_test_indian.csv
    test_path = "data/own_test_indian.csv"
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        print(f"Scanning {len(df_test)} transactions in {test_path} for PII...")
        
        # Regex for personal UPI handles (e.g. personal names like deep@okhdfc, 9876543210@ybl)
        vpa_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
        phone_pattern = re.compile(r'\b[6-9]\d{9}\b')
        acc_pattern = re.compile(r'\b\d{11,16}\b')
        
        pii_found = []
        for idx, row in df_test.iterrows():
            desc = str(row["transaction_description"])
            vpas = vpa_pattern.findall(desc)
            phones = phone_pattern.findall(desc)
            accs = acc_pattern.findall(desc)
            
            # Filter out known commercial handles
            commercial_handles = {"@icici", "@hdfcbank", "@apl", "@axisbank", "@paytm"}
            personal_vpas = [v for v in vpas if not any(v.endswith(h) for h in commercial_handles)]
            
            if personal_vpas or phones or accs:
                pii_found.append((idx + 2, desc, personal_vpas, phones, accs))
                
        print(f"Potential PII issues flagged: {len(pii_found)}")
        for line_no, desc, vpas, phones, accs in pii_found:
            print(f"  Line {line_no}: {desc}")
            if vpas: print(f"    Personal VPA: {vpas}")
            if phones: print(f"    Phone: {phones}")
            if accs: print(f"    Account/Long Digits: {accs}")
            
    # Secrets check in repo
    print("\nScanning repository for secrets / .env...")
    for root, dirs, files in os.walk("."):
        if ".git" in root or ".venv" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith((".env", ".key", ".pem")) or f == ".env":
                full_path = os.path.join(root, f)
                print(f"  Found potential secrets file: {full_path}")
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as sf:
                        for line in sf:
                            if any(k in line.upper() for k in ["API_KEY", "SECRET", "PASSWORD", "TOKEN", "BEARER"]):
                                masked = re.sub(r'(=|:)\s*(.{4}).*', r'\1 \2****', line.strip())
                                print(f"    [SECRET MATCH in {full_path}]: {masked}")
                except Exception as e:
                    print(f"    Could not read {full_path}: {e}")

if __name__ == "__main__":
    run_merchant_overlap_check()
    run_noise_traceability()
    run_xgboost_alignment_check()
    run_security_pii_check()
