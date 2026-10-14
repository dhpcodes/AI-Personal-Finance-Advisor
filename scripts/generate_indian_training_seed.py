"""Generate Leakage-Safe Indian Merchant Training Seed Dataset.

Strict Leakage Safeguards:
- Asserts ZERO overlap with held-out data/own_test_indian.csv.
- Generates diverse, realistic synthetic Indian banking descriptions.
- Covers the 10 standard category taxonomies with balanced representation.
- Saves to data/indian_merchants_train_seed.csv.
"""

import sys
import random
import re
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OWN_TEST_PATH = REPO_ROOT / "data" / "own_test_indian.csv"
OUTPUT_SEED_PATH = REPO_ROOT / "data" / "indian_merchants_train_seed.csv"

# Load held-out test descriptions to guarantee zero overlap
assert OWN_TEST_PATH.exists(), f"Missing {OWN_TEST_PATH}"
own_test_df = pd.read_csv(OWN_TEST_PATH)
HELD_OUT_DESCS = set(own_test_df["transaction_description"].str.strip().str.upper())

print(f"Loaded {len(HELD_OUT_DESCS)} held-out test descriptions to prevent leakage.")

# Seed definitions for Indian merchants across the 10 classes
INDIAN_MERCHANT_TEMPLATES = {
    "Food & Dining": [
        "SWIGGY", "ZOMATO", "BLINKIT", "ZEPTO", "BIGBASKET",
        "DOMINOS PIZZA", "PIZZA HUT", "KFC INDIA", "MCDONALDS INDIA",
        "BURGER KING INDIA", "CHAAYOS", "CHAI POINT", "CAFE COFFEE DAY",
        "BARBEQUE NATION", "HALDIRAMS", "BIKANERVALA", "PARAS DAIRY",
        "AMUL ICE CREAM", "NATURAL ICE CREAM", "PUNJABI GRILL",
        "SAGAR RATNA", "SARAVANAA BHAVAN", "BEHROUZ BIRYANI", "FAASOS",
        "OVEN STORY PIZZA", "MAD OVER DONUTS", "THEOBROMA", "WOW MOMO"
    ],
    "Transportation": [
        "OLA CABS", "OLA AUTO", "UBER INDIA", "RAPIDO BIKE",
        "BLUSMART CAB", "IRCTC TRAIN", "MAKEMYTRIP", "EASEMYTRIP",
        "YATRA TRAVELS", "INDIGO AIRLINES", "AIR INDIA", "SPICEJET",
        "AKASA AIR", "NHAI FASTAG", "PAYTM FASTAG", "ICICI FASTAG",
        "INDIAN OIL PETROL", "BHARAT PETROLEUM", "HINDUSTAN PETROLEUM",
        "SHELL INDIA FUEL", "BEST BUS MUMBAI", "DMRC METRO DELHI",
        "BMRCL METRO BANGALORE", "AUTO FARE METRED", "CHALO APP PASS"
    ],
    "Shopping & Retail": [
        "AMAZON PAY", "AMAZON INDIA", "FLIPKART INTERNET", "MYNTRA DESIGNS",
        "AJIO TRENDS", "NYKAA E-RETAIL", "TATA CLIQ", "MEESHO ONLINE",
        "LENSKART", "FIRSTCRY BABY", "PURPLLE COSMETICS", "SNAPDEAL",
        "RELIANCE DIGITAL", "CROMA ELECTRONICS", "VIJAY SALES",
        "DECATHLON SPORTS", "ZARA INDIA", "H&M RETAIL", "UNIQLO INDIA",
        "WESTSIDE TATA", "PANTALOONS FASHION", "MAX FASHION",
        "DMART SUPERMART", "SPENCERS RETAIL", "MORE SUPERMARKET",
        "LIFESTYLE RETAIL", "SHOPPERS STOP", "TITAN EYEPLUS", "TANISHQ JEWELLERY"
    ],
    "Utilities & Services": [
        "RELIANCE JIO", "JIO PREPAID", "JIO FIBER", "BHARTI AIRTEL",
        "AIRTEL BROADBAND", "VODAFONE IDEA", "VI RECHARGE", "MTNL DELHI",
        "BSNL PREPAID", "TATA POWER", "ADANI ELECTRICITY", "BESCOM BANGALORE",
        "MSEDCL MAHAVITARAN", "UPPCL POWER", "CESC KOLKATA", "TSSPDCL HYDERABAD",
        "IGL INDRAPRASTHA GAS", "MGL MAHANAGAR GAS", "GUJARAT GAS", "ADANI TOTAL GAS",
        "ACT FIBERNET", "HATHWAY BROADBAND", "TATA PLAY DTH", "DISH TV RECHARGE",
        "URBAN COMPANY", "URBANCLAP SERVICES", "HOUSEJOY HOME CLEANING"
    ],
    "Financial Services": [
        "GROWW INVESTECH", "ZERODHA BROKING", "UPSTOX SECURITIES",
        "ANGEL ONE BROKING", "PAYTM MONEY", "5PAISA CAPITAL", "KOTAK SECURITIES",
        "MOTILAL OSWAL", "HDFC SECURITIES", "ICICI DIRECT", "SHAREKHAN TRADING",
        "LIC OF INDIA", "HDFC LIFE INSURANCE", "SBI LIFE INSURANCE",
        "ICICI PRUDENTIAL LIFE", "MAX LIFE INSURANCE", "BAJAJ ALLIANZ LIFE",
        "STAR HEALTH INSURANCE", "CARE HEALTH INSURANCE", "NIPPON INDIA MF",
        "SBI MUTUAL FUND", "HDFC MUTUAL FUND", "ICICI PRUDENTIAL MF"
    ],
    "Entertainment & Recreation": [
        "DISNEY+ HOTSTAR", "SONY LIV SUBSCRIPTION", "ZEE5 ENTERTAINMENT",
        "NETFLIX INDIA", "AMAZON PRIME VIDEO", "SPOTIFY INDIA", "JIOSAAVN MUSIC",
        "GAANA PREMIUM", "WYNK MUSIC", "BOOKMYSHOW", "PAYTM INSIDER TICKETS",
        "PVR CINEMAS", "INOX LEISURE", "CINEPOLIS MULTIPLEX", "SMAASH GAMING",
        "TIMEZONE GAMES", "WONDERLA HOLIDAYS", "IMAGICA WORLD", "KIDZANIA INDIA"
    ],
    "Healthcare & Medical": [
        "APOLLO PHARMACY", "TATA 1MG", "PHARMEASY", "NETMEDS MEDICINES",
        "MEDPLUS HEALTH", "WELLNESS FOREVER", "DR LAL PATHLABS",
        "METROPOLIS HEALTHCARE", "THYROCARE DIAGNOSTICS", "SRL DIAGNOSTICS",
        "FORTIS HEALTHCARE", "MAX HOSPITAL CLINIC", "MANIPAL HOSPITALS",
        "APOLLO HOSPITALS", "NARAYANA HEALTH", "MEDANTA THE MEDICITY",
        "CLOVE DENTAL", "VASAN EYE CARE", "CARE HOSPITALS", "LILAVATI HOSPITAL"
    ],
    "Government & Legal": [
        "INCOME TAX DEPT", "TIN NSDL TAX", "GST PORTAL PAYMENT",
        "MCD PROPERTY TAX", "BMC PROPERTY TAX", "BBMP PROPERTY TAX",
        "DELHI TRAFFIC CHALLAN", "MAHARASHTRA E-CHALLAN", "BANGALORE TRAFFIC POLICE",
        "PASSPORT SEVA KENDRA", "RTO DRIVING LICENSE", "RTO VEHICLE TAX",
        "SUB REGISTRAR STAMP DUTY", "DISTRICT COURT FEES", "HIGH COURT REGISTRY",
        "MUNICIPAL WATER TAX", "CONSUMER COURT FEES"
    ],
    "Charity & Donations": [
        "PM CARES FUND", "PRADHAN MANTRI RELIEF", "CRY CHILD RIGHTS",
        "GOONJ RELIEF NGO", "AKSHAYA PATRA FOUNDATION", "HELPAGE INDIA",
        "SMILE FOUNDATION", "NANHI KALI DONATION", "CRY INDIA NGO",
        "UNICEF INDIA DONATION", "SAVE THE CHILDREN INDIA", "RAMAKRISHNA MISSION",
        "ISCKON MIDDAY MEAL", "AMRITA VISHWA VIDYAPEETHAM CHARITY"
    ],
    "Income": [
        "SALARY CREDIT INFOSYS", "SALARY CREDIT TCS", "SALARY CREDIT WIPRO",
        "SALARY CREDIT HCL TECH", "SALARY CREDIT ACCENTURE", "SALARY CREDIT TECH MAHINDRA",
        "SALARY CREDIT COGNIZANT", "SALARY CREDIT LTI MINDTREE", "MONTHLY SALARY CREDIT",
        "PAYROLL DIRECT DEPOSIT", "CONSULTING FEES CREDIT", "FREELANCE UPWORK ESCROW",
        "DIVIDEND ECS CREDIT", "SAVINGS BANK INTEREST CREDIT", "INCOME TAX REFUND CPC",
        "BONUS INCENTIVE EMPLOYER", "CLIENT DIRECT REMITTANCE", "COMMISSION PAYMENT CREDIT"
    ]
}

CITIES = ["MUMBAI", "DELHI", "BANGALORE", "CHENNAI", "HYDERABAD", "PUNE", "KOLKATA", "GURGAON", "NOIDA", "AHMEDABAD", "JAIPUR"]
BANKS = ["icici", "hdfcbank", "axisbank", "sbi", "kotak", "paytm", "okaxis", "okhdfcbank", "okicici"]
CHANNELS = ["ONLINE", "STORE", "RETAIL", "APP", "PORTAL", "TXN", "BRANCH"]


def generate_variations(merchant: str, rng: random.Random) -> list:
    """Generate realistic variations with diverse banking channels."""
    vpa = re.sub(r"[^a-z]", "", merchant.lower())[:8] or "merch"
    bank = rng.choice(BANKS)
    city = rng.choice(CITIES)
    ref12 = rng.randint(10**11, 10**12 - 1)
    ref6 = rng.randint(10**5, 10**6 - 1)
    card4 = rng.randint(1000, 9999)
    card4_2 = rng.randint(1000, 9999)

    variations = [
        # UPI forms
        f"UPI/{ref12}/{merchant}/{vpa}@{bank}",
        f"UPI/{ref12}/{merchant}/{merchant.lower().replace(' ', '')}@{bank}",
        f"UPI/{merchant}/{vpa}@{bank}",
        
        # POS slip forms
        f"POS {card4}XXXX{card4_2} {merchant} {city}",
        f"POS {card4}XXXX{card4_2} {merchant}",
        
        # NEFT / RTGS / IMPS forms
        f"NEFT-{ref12}-{merchant}",
        f"NEFT-{ref6}-{merchant} {city}",
        f"IMPS/{ref6}/{merchant}/REF{rng.randint(100, 999)}",
        
        # Channel / Retail formats
        f"{merchant} {rng.choice(CHANNELS)} {city}",
        f"{merchant} #{rng.randint(1000, 9999)}",
        f"{merchant} {rng.choice(CHANNELS)}",
        f"{merchant} ORDER TXN{ref6}",
        f"{merchant} {city}",
        f"{merchant}"
    ]
    return variations


def main():
    rng = random.Random(42)
    records = []
    skipped_leakage_count = 0

    print("Generating synthetic Indian merchant dataset...")
    for category, merchants in INDIAN_MERCHANT_TEMPLATES.items():
        for m in merchants:
            variations = generate_variations(m, rng)
            for v in variations:
                v_clean = v.strip().upper()
                # Strict Leakage Assertion: Must NEVER match any held-out test string
                if v_clean in HELD_OUT_DESCS:
                    skipped_leakage_count += 1
                    continue
                records.append({
                    "transaction_description": v,
                    "category": category,
                    "core": m.lower(),
                    "is_indian_seed": True
                })

    df_seed = pd.DataFrame(records).drop_duplicates(subset=["transaction_description"]).reset_index(drop=True)
    
    # Shuffle reproducibility
    df_seed = df_seed.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\nGenerated {len(df_seed):,} unique Indian merchant seed records.")
    print(f"Skipped {skipped_leakage_count} descriptions due to exact match with test set (Leakage Prevention).")
    print("\nClass distribution:")
    print(df_seed["category"].value_counts())

    # Final assertion
    overlap = set(df_seed["transaction_description"].str.strip().str.upper()).intersection(HELD_OUT_DESCS)
    assert len(overlap) == 0, f"FATAL: Overlap detected with held-out test set: {overlap}"
    print("\n[VERIFIED] Zero overlap with held-out test set: len(overlap) == 0.")

    # Save artifact
    OUTPUT_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_seed.to_csv(OUTPUT_SEED_PATH, index=False)
    print(f"Saved seed dataset to {OUTPUT_SEED_PATH}")


if __name__ == "__main__":
    main()
