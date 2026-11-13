"""Light-weight slot extraction from the (already PII-masked) customer message."""

import re

_AMOUNT = re.compile(
    r"(?:(?P<cur1>£|\$|€|₹|gbp|usd|eur|inr)\s?(?P<num1>\d[\d,]*(?:\.\d{1,2})?)\s?(?P<k1>k)?)"
    r"|(?:(?P<num2>\d[\d,]*(?:\.\d{1,2})?)\s?(?P<k2>k)?\s?(?P<cur2>pounds?|quid|dollars?|euros?|rupees?|gbp|usd|eur|inr))",
    re.I,
)
_TENURE = re.compile(r"(\d{1,3})\s*(years?|yrs?|months?|mos?)\b", re.I)
_LOAN_TYPE = re.compile(r"\b(personal|car|auto|vehicle|home|house|mortgage|student|education)\b", re.I)
_MERCHANT = re.compile(r"\b(?:at|from|to)\s+([A-Z][\w&'-]+)")

_CURRENCY = {"£": "GBP", "$": "USD", "€": "EUR", "₹": "INR", "pound": "GBP", "quid": "GBP", "dollar": "USD",
             "euro": "EUR", "rupee": "INR"}
_LOAN_MAP = {"auto": "car", "vehicle": "car", "house": "home", "mortgage": "home", "education": "student"}


def _currency(raw: str) -> str:
    raw = raw.lower().rstrip("s")
    return _CURRENCY.get(raw, raw.upper())


def extract_entities(text: str) -> dict:
    out: dict = {}
    m = _AMOUNT.search(text)
    if m:
        num = m.group("num1") or m.group("num2")
        value = float(num.replace(",", ""))
        if m.group("k1") or m.group("k2"):
            value *= 1000
        out["amount"] = value
        out["currency"] = _currency(m.group("cur1") or m.group("cur2"))
    t = _TENURE.search(text)
    if t:
        n = int(t.group(1))
        out["tenure_months"] = n * 12 if t.group(2).lower().startswith("y") else n
    lt = _LOAN_TYPE.search(text)
    if lt:
        word = lt.group(1).lower()
        out["loan_type"] = _LOAN_MAP.get(word, word)
    mm = _MERCHANT.search(text)
    if mm and not mm.group(1).startswith("["):
        out["merchant"] = mm.group(1)
    return out
