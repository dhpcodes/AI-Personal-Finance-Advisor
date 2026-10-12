import re
from typing import Dict, Any, List, Set

class PrivacyViolationError(Exception):
    """Raised when an outbound payload contains unauthorized or sensitive financial data."""
    pass

class PrivacyGuard:
    """
    Outbound data minimization and privacy assurance firewall.
    Ensures that external LLM API calls NEVER receive raw transaction records,
    account numbers, passwords, card details, or unapproved metadata.
    """

    # Explicit allowlist of permissible fields in outbound LLM payloads
    ALLOWED_SUMMARY_KEYS: Set[str] = {
        "status",
        "message",
        "intent",
        "question",
        "currency",
        "category",
        "total_income",
        "total_expense",
        "total_spending",
        "net_cash_flow",
        "highest_spending_category",
        "highest_spending_amount",
        "category_spending",
        "category_percentages",
        "category_breakdown",
        "monthly_totals",
        "monthly_breakdown",
        "recurring_expenses_count",
        "recurring_expenses",
        "recurring_summary",
        "merchant",
        "average_amount",
        "frequency",
        "anomaly_count",
        "anomalies",
        "anomaly_summary",
        "forecast_point",
        "forecast_lower_bound",
        "forecast_upper_bound",
        "forecast_category_breakdown",
        "confidence",
        "predicted_category",
        "top_words",
        "target_amount",
        "months",
        "required_monthly_savings",
        "current_surplus",
        "feasibility",
        "method",
        "explanation",
        "summary"
    }

    # Prohibited patterns indicating unmasked sensitive financial PII
    FORBIDDEN_PATTERNS = [
        re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"), # 16-digit card number
        re.compile(r"\b\d{9,18}\b"),                             # Bank account number
        re.compile(r"\b(password|pin|cvv|cvc|otp)\b", re.I),    # Credentials
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                  # Indian PAN number
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),                 # Aadhaar format
        re.compile(r"\b(?:(?:\+91|91|0)[ -]?)?[6789]\d{9}\b")   # Indian 10-digit mobile number
    ]

    @classmethod
    def validate_outbound_payload(cls, payload: Dict[str, Any]) -> bool:
        """
        Validates whether an outbound dictionary complies with the strict allowlist
        and contains no prohibited sensitive PII.
        """
        if not isinstance(payload, dict):
            return False

        # 1. Reject full raw transaction arrays
        for k in ["transactions", "raw_transactions", "user_df", "all_records"]:
            if k in payload and isinstance(payload[k], list) and len(payload[k]) > 0:
                raise PrivacyViolationError(f"Data Minimization Violation: Outbound payload contains raw transaction list '{k}'.")

        # 2. Check keys against allowlist
        for key, value in payload.items():
            if key not in cls.ALLOWED_SUMMARY_KEYS:
                # Disallow arbitrary custom metadata keys
                raise PrivacyViolationError(f"Field '{key}' is not on the approved outbound allowlist.")

            # 3. Check for sensitive PII in strings
            if isinstance(value, str):
                for pattern in cls.FORBIDDEN_PATTERNS:
                    if pattern.search(value):
                        raise PrivacyViolationError(f"Potential credential or account PII detected in field '{key}'.")

        return True

    @classmethod
    def sanitize_outbound_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filters an outbound dictionary down to strictly allowlisted fields
        and redacts any suspicious substrings.
        """
        if not isinstance(payload, dict):
            return {}

        sanitized: Dict[str, Any] = {}
        for k, v in payload.items():
            if k in cls.ALLOWED_SUMMARY_KEYS:
                if isinstance(v, str):
                    clean_str = v
                    for pattern in cls.FORBIDDEN_PATTERNS:
                        clean_str = pattern.sub("[REDACTED_PII]", clean_str)
                    sanitized[k] = clean_str
                elif isinstance(v, (int, float, bool)):
                    sanitized[k] = v
                elif isinstance(v, dict):
                    # Sanitize nested dictionaries (e.g. category spending breakdown)
                    sanitized[k] = {
                        sub_k: sub_v for sub_k, sub_v in v.items()
                        if isinstance(sub_v, (int, float, str, bool))
                    }
                elif isinstance(v, list):
                    # Only allow lists of simple scalars or summary dicts
                    sanitized[k] = [
                        item for item in v
                        if isinstance(item, (int, float, str))
                    ]

        return sanitized
