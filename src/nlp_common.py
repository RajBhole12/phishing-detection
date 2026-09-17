"""
nlp_common.py
--------------
Shared, lightweight text-analysis utilities for the NEW Email and Message
phishing detectors. Completely independent of the existing URL feature
extraction module (src/features/url_features.py), which is untouched.

Deliberately avoids a hard dependency on NLTK (which would require
downloading corpora at runtime / network access to work) in favor of
scikit-learn's built-in English stopword list inside TfidfVectorizer.
This keeps installation lightweight and consistent with the rest of the
project's "install and run" experience.
"""

import re

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

SHORTENER_DOMAINS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
                      "is.gd", "cutt.ly", "rb.gy", "buff.ly", "shorturl.at"]

SUSPICIOUS_WORDS = [
    "verify", "suspend", "suspended", "confirm", "update your", "account",
    "security alert", "unusual activity", "click here", "act now",
    "limited time", "restricted", "unauthorized", "validate", "locked",
    "unlock your", "reactivate", "important notice",
]

URGENCY_WORDS = [
    "urgent", "immediately", "asap", "right away", "act now", "expire",
    "expires", "expiring", "deadline", "final notice", "last chance",
    "within 24 hours", "today only", "response required", "time sensitive",
]

OTP_WORDS = [
    "otp", "one-time password", "one time password", "verification code",
    "security code", "pin code", "passcode", "authentication code",
]

PASSWORD_WORDS = [
    "password", "reset your password", "login credentials",
    "username and password", "confirm your password", "update your password",
]

PAYMENT_WORDS = [
    "payment", "invoice", "bank transfer", "wire transfer", "credit card",
    "gift card", "bitcoin", "crypto", "refund", "billing", "outstanding balance",
]

PERSONAL_INFO_WORDS = [
    "social security", "ssn", "date of birth", "credit card number",
    "bank account number", "passport number", "national id", "mother's maiden name",
]

PRIZE_WORDS = [
    "congratulations", "winner", "you have won", "you've won", "claim your prize",
    "free gift", "lottery", "reward", "cash prize", "selected", "giveaway",
]

VERIFICATION_WORDS = [
    "verify your account", "confirm your identity", "account verification",
    "verify your identity", "unusual sign-in", "suspicious login",
]


def clean_text(text: str) -> str:
    return (text or "").strip()


def count_keyword_hits(text_lower: str, keywords) -> int:
    return sum(1 for kw in keywords if kw in text_lower)


def extract_urls(text: str) -> list:
    return URL_REGEX.findall(text or "")


def has_shortener(urls: list) -> bool:
    return any(any(s in u.lower() for s in SHORTENER_DOMAINS) for u in urls)


def caps_ratio(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return round(caps / len(letters), 4)


def excessive_punctuation_count(text: str) -> int:
    return len(re.findall(r"[!?]{2,}", text or ""))


def exclamation_count(text: str) -> int:
    return (text or "").count("!")
