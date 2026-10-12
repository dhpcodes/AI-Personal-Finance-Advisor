"""Transaction description canonicalization module for Indian bank statements.

Extracts normalized merchant names by stripping standard Indian banking wrappers:
- UPI VPA wrappers (UPI/ref/MERCHANT/vpa@bank)
- POS slip wrappers and masked card identifiers (POS 4129XXXX9812 MERCHANT CITY)
- NEFT / RTGS prefixes (NEFT-ref-MERCHANT)
- IMPS prefixes and trailing reference numbers (IMPS/ref/MERCHANT/REF123)
- Masked card tokens and long numeric authorization references
"""

import re
from typing import Union, List, Sequence
import pandas as pd


# Pre-compiled canonicalization regexes
_RE_UPI = re.compile(r"^UPI(?:/[0-9]+)?/([^/]+)(?:/[a-z0-9._]+@[a-z0-9]+)?$", re.IGNORECASE)
_RE_UPI_FALLBACK = re.compile(r"\bUPI/[0-9]+/|/[a-z0-9._]+@[a-z0-9]+\b|@[a-z0-9]+\b", re.IGNORECASE)

_RE_NEFT = re.compile(r"^NEFT-[0-9]+-(.+)$", re.IGNORECASE)
_RE_IMPS = re.compile(r"^IMPS/[0-9]+/([^/]+)(?:/REF[0-9]+)?$", re.IGNORECASE)

_RE_POS = re.compile(r"^POS\s+[0-9]{2,6}X{2,12}[0-9]{0,6}\s+(.+?)(?:\s+(?:MUMBAI|DELHI|BANGALORE|CHENNAI|HYDERABAD|KOLKATA|PUNE|AHMEDABAD|JAIPUR|SURAT|GURGAON|NOIDA))?$", re.IGNORECASE)

_RE_MASKED_CARDS = re.compile(r"\b[0-9]{0,6}X{2,12}[0-9]{0,6}\b", re.IGNORECASE)
_RE_LONG_NUMS = re.compile(r"\b[0-9]{8,16}\b")
_RE_REF_CODE = re.compile(r"/REF[0-9]+\b", re.IGNORECASE)
_RE_TRUNC_ELLIPSIS = re.compile(r"\.{2,}$")
_RE_EXTRA_SPACES = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    """Canonicalize a transaction description string.

    Strips UPI, POS, NEFT, IMPS wrappers, card numbers, authorization codes,
    and trailing truncation indicators while preserving merchant identity tokens.
    """
    if not isinstance(text, str):
        return ""

    s = text.strip()
    if not s:
        return ""

    # 1. Exact wrapper matching
    m_upi = _RE_UPI.match(s)
    if m_upi:
        s = m_upi.group(1).strip()
    else:
        m_neft = _RE_NEFT.match(s)
        if m_neft:
            s = m_neft.group(1).strip()
        else:
            m_imps = _RE_IMPS.match(s)
            if m_imps:
                s = m_imps.group(1).strip()
            else:
                m_pos = _RE_POS.match(s)
                if m_pos:
                    s = m_pos.group(1).strip()

    # 2. General token-level cleanup
    s = _RE_UPI_FALLBACK.sub(" ", s)
    s = _RE_MASKED_CARDS.sub(" ", s)
    s = _RE_LONG_NUMS.sub(" ", s)
    s = _RE_REF_CODE.sub(" ", s)
    s = _RE_TRUNC_ELLIPSIS.sub("", s)

    # 3. Collapse whitespace and strip
    s = _RE_EXTRA_SPACES.sub(" ", s).strip()

    # If all tokens were stripped, return original non-empty trimmed text
    return s if s else text.strip()


def canonicalize_series(texts: Sequence[str], raw: bool = False) -> List[str]:
    """Process a sequence of transaction descriptions.

    Args:
        texts: Iterable of raw strings.
        raw: If True, bypasses canonicalization and returns stripped raw text.

    Returns:
        List of processed transaction descriptions.
    """
    if raw:
        return [str(t).strip() for t in texts]
    return [canonicalize(t) for t in texts]
