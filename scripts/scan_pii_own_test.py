"""Scan data/own_test_indian.csv for PII (Personal Data, Phone Numbers, Account Numbers, P2P UPI IDs).

Outputs:
- Generates a report of any potential PII.
- Creates a sanitized copy at data/own_test_indian_sanitized.csv with placeholders.
- Leaves the original data/own_test_indian.csv unmodified.
"""

import re
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "own_test_indian.csv"
SANITIZED_PATH = REPO_ROOT / "data" / "own_test_indian_sanitized.csv"

# Recognized merchant handles vs likely personal P2P handles
KNOWN_MERCHANT_VPA_WORDS = {
    "swiggy", "zomato", "amazon", "apl", "flipkart", "ola", "olacabs", "jio", "paytm",
    "uber", "dominos", "airtel", "tatasky", "actcorp", "bescom", "netflix", "spotify",
    "bigbasket", "zepto", "blinkit", "grofers", "dunzo", "myntra", "nykaa", "irctc",
    "makemytrip", "yatra", "cleartrip", "bookmyshow", "pvr", "inox", "cultfit", "cred",
    "apollo", "pharmeasy", "1mg", "netmeds", "starbucks", "mcdonalds", "kfc", "subway",
    "pizza", "burger", "retail", "supermarket", "store", "mart", "bazaar", "shop"
}

def scan_for_pii(text: str) -> list[str]:
    flags = []
    
    # 1. Phone numbers (10 digits starting with 6, 7, 8, 9, or prefixed by +91 / 91)
    phone_match = re.findall(r"(?:(?:\+?91)[\s-]?)?[6-9]\d{9}\b", text)
    if phone_match:
        flags.append(f"Phone Number: {phone_match}")
        
    # 2. Account numbers (standalone sequences of 10 to 18 digits that are not timestamps/standard 12-digit UPI RRNs)
    # UPI reference numbers (RRN) are typically 12 digits e.g. UPI/308942189012/
    raw_digits = re.findall(r"\b\d{10,18}\b", text)
    for d in raw_digits:
        # If it's part of UPI/12digits/, it's an RRN token, but still check if it looks like an account number
        if len(d) in [11, 14, 15, 16] and not text.startswith("UPI/"):
            flags.append(f"Possible Account Number: {d}")

    # 3. P2P UPI handles (e.g. rahul.sharma@okaxis, 9876543210@paytm, john123@ybl)
    upi_handles = re.findall(r"([a-zA-Z0-9_\.\-]+)@([a-zA-Z0-9]+)", text)
    for handle, bank in upi_handles:
        handle_lower = handle.lower()
        # If it's not a known brand/merchant keyword
        if not any(m in handle_lower for m in KNOWN_MERCHANT_VPA_WORDS):
            # Check if it has personal name characteristics (dots, digits, common personal names)
            if re.match(r"^[a-z]+\.[a-z]+", handle_lower) or re.match(r"^[6-9]\d{9}$", handle_lower) or len(handle_lower) < 15:
                flags.append(f"Potential Personal P2P UPI VPA: {handle}@{bank}")

    return flags

def sanitize_text(text: str) -> str:
    s = text
    # Sanitize 10-digit phone numbers
    s = re.sub(r"\b([6-9]\d{2})\d{4}(\d{3})\b", r"\1XXXX\2", s)
    
    # Sanitize P2P UPI handles (replace name part with user_placeholder)
    def vpa_sub(match):
        handle = match.group(1)
        bank = match.group(2)
        if any(m in handle.lower() for m in KNOWN_MERCHANT_VPA_WORDS):
            return match.group(0)
        return f"user_placeholder@{bank}"
        
    s = re.sub(r"\b([a-zA-Z0-9_\.\-]+)@([a-zA-Z0-9]+)\b", vpa_sub, s)
    return s

def main():
    print("=" * 70)
    print("AUDIT: PII SCAN IN data/own_test_indian.csv")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"Total rows in own_test_indian.csv: {len(df)}")

    findings = []
    sanitized_descriptions = []

    for idx, row in df.iterrows():
        desc = str(row["transaction_description"])
        pii_flags = scan_for_pii(desc)
        if pii_flags:
            findings.append((idx + 1, desc, pii_flags))
        sanitized_descriptions.append(sanitize_text(desc))

    print(f"\nTotal potential PII items detected: {len(findings)}")
    for row_num, desc, flags in findings:
        print(f"  Row {row_num:3d}: {desc}")
        for flag in flags:
            print(f"           -> {flag}")

    # Write sanitized copy
    df_sanitized = df.copy()
    df_sanitized["transaction_description"] = sanitized_descriptions
    df_sanitized.to_csv(SANITIZED_PATH, index=False)
    print(f"\n[SAVED] Created sanitized copy at: {SANITIZED_PATH.relative_to(REPO_ROOT)}")
    print("NOTE: Original data/own_test_indian.csv has been left 100% UNTOUCHED.")

if __name__ == "__main__":
    main()
