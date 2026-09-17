"""
email_features.py
-------------------
Feature extraction for the NEW Email Phishing Detection feature.
Completely independent of src/features/url_features.py (the existing,
untouched Website/URL feature extractor).

Only analyzes text the user pastes in manually -- never opens links,
never accesses a real mailbox.
"""

import re

from src.nlp_common import (
    clean_text, count_keyword_hits, extract_urls, has_shortener, caps_ratio,
    excessive_punctuation_count, exclamation_count,
    SUSPICIOUS_WORDS, URGENCY_WORDS, OTP_WORDS, PASSWORD_WORDS,
    PAYMENT_WORDS, PERSONAL_INFO_WORDS, VERIFICATION_WORDS,
)

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "live.com", "gmx.com",
}

FEATURE_ORDER = [
    "email_length", "subject_length", "num_suspicious_words", "num_urgency_words",
    "has_otp_request", "has_password_request", "has_payment_request",
    "has_personal_info_request", "has_verification_request", "num_links",
    "has_link_shortener", "sender_domain_is_free_provider", "sender_missing_domain",
    "caps_ratio", "excessive_punctuation", "exclamation_count",
]

FEATURE_RISK_WEIGHTS = {
    "has_otp_request": 22,
    "has_password_request": 20,
    "has_payment_request": 18,
    "has_personal_info_request": 22,
    "has_verification_request": 14,
    "has_link_shortener": 16,
    "sender_missing_domain": 10,
}


def get_sender_domain(sender_email: str) -> str:
    m = re.search(r"@([\w.-]+)", sender_email or "")
    return m.group(1).lower() if m else ""


def extract_email_features(sender: str, subject: str, body: str):
    """Returns (feature_dict, combined_text_for_ml)."""
    sender = clean_text(sender)
    subject = clean_text(subject)
    body = clean_text(body)

    full_text = f"{subject}\n{body}"
    full_lower = full_text.lower()
    domain = get_sender_domain(sender)
    urls = extract_urls(body) + extract_urls(subject)

    feats = {
        "email_length": len(body),
        "subject_length": len(subject),
        "num_suspicious_words": count_keyword_hits(full_lower, SUSPICIOUS_WORDS),
        "num_urgency_words": count_keyword_hits(full_lower, URGENCY_WORDS),
        "has_otp_request": int(count_keyword_hits(full_lower, OTP_WORDS) > 0),
        "has_password_request": int(count_keyword_hits(full_lower, PASSWORD_WORDS) > 0),
        "has_payment_request": int(count_keyword_hits(full_lower, PAYMENT_WORDS) > 0),
        "has_personal_info_request": int(count_keyword_hits(full_lower, PERSONAL_INFO_WORDS) > 0),
        "has_verification_request": int(count_keyword_hits(full_lower, VERIFICATION_WORDS) > 0),
        "num_links": len(urls),
        "has_link_shortener": int(has_shortener(urls)),
        "sender_domain_is_free_provider": int(domain in FREE_EMAIL_DOMAINS),
        "sender_missing_domain": int(domain == ""),
        "caps_ratio": caps_ratio(full_text),
        "excessive_punctuation": excessive_punctuation_count(full_text),
        "exclamation_count": exclamation_count(full_text),
    }
    return feats, full_text


def compute_email_feature_risk_score(feats: dict) -> float:
    """Transparent, additive heuristic risk score (0-100) from email features."""
    score = 0.0
    for key, weight in FEATURE_RISK_WEIGHTS.items():
        if feats.get(key):
            score += weight

    score += min(20, feats.get("num_urgency_words", 0) * 7)
    score += min(18, feats.get("num_suspicious_words", 0) * 4)

    n_links = feats.get("num_links", 0)
    if n_links >= 3:
        score += 14
    elif n_links >= 1:
        score += 6

    if feats.get("caps_ratio", 0) > 0.3:
        score += 8
    if feats.get("excessive_punctuation", 0) >= 2:
        score += 6
    if feats.get("exclamation_count", 0) >= 3:
        score += 5

    return round(min(100.0, score), 2)


def explain_email_features(feats: dict) -> list:
    """Human-readable risk factors, same shape as the URL scanner's explanations."""
    reasons = []

    if feats["has_otp_request"]:
        reasons.append({"level": "high", "text": "Requests an OTP / verification code"})
    if feats["has_password_request"]:
        reasons.append({"level": "high", "text": "Requests a password or login credentials"})
    if feats["has_personal_info_request"]:
        reasons.append({"level": "high", "text": "Requests sensitive personal information"})
    if feats["has_payment_request"]:
        reasons.append({"level": "high", "text": "Requests a payment, transfer, or billing action"})
    if feats["has_verification_request"]:
        reasons.append({"level": "medium", "text": "Asks the recipient to verify their account or identity"})
    if feats["has_link_shortener"]:
        reasons.append({"level": "high", "text": "Contains a shortened URL (hides real destination)"})
    if feats["num_links"] >= 3:
        reasons.append({"level": "medium", "text": "Contains multiple links"})
    elif feats["num_links"] >= 1:
        reasons.append({"level": "low", "text": "Contains a link"})
    if feats["num_urgency_words"] >= 2:
        reasons.append({"level": "high", "text": "Multiple urgency-related phrases detected"})
    elif feats["num_urgency_words"] == 1:
        reasons.append({"level": "medium", "text": "Urgency-related language detected"})
    if feats["num_suspicious_words"] >= 2:
        reasons.append({"level": "medium", "text": "Multiple suspicious phrases detected"})
    if feats["sender_missing_domain"]:
        reasons.append({"level": "medium", "text": "Sender address has no identifiable domain"})
    if feats["caps_ratio"] > 0.3:
        reasons.append({"level": "medium", "text": "Excessive use of capital letters"})
    if feats["excessive_punctuation"] >= 2:
        reasons.append({"level": "low", "text": "Excessive punctuation (e.g. '!!!', '??')"})

    if not reasons:
        reasons.append({"level": "safe", "text": "No major risk indicators found in the email content"})

    return reasons
