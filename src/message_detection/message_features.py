"""
message_features.py
---------------------
Feature extraction for the NEW Message (SMS/chat) Phishing Detection
feature. Independent of src/features/url_features.py and of
src/email_detection/ (no sender/subject fields here -- just message text).

Only analyzes text the user pastes in manually -- never accesses a real
SMS inbox, messaging app, or contacts.
"""

from src.nlp_common import (
    clean_text, count_keyword_hits, extract_urls, has_shortener, caps_ratio,
    excessive_punctuation_count, exclamation_count,
    URGENCY_WORDS, OTP_WORDS, PASSWORD_WORDS, PAYMENT_WORDS, PRIZE_WORDS,
    VERIFICATION_WORDS,
)

FEATURE_ORDER = [
    "message_length", "num_urgency_words", "num_prize_words",
    "has_verification_request", "has_payment_request", "has_otp_request",
    "has_password_request", "num_links", "has_link_shortener",
    "caps_ratio", "excessive_punctuation", "exclamation_count",
]

FEATURE_RISK_WEIGHTS = {
    "has_otp_request": 22,
    "has_password_request": 18,
    "has_payment_request": 18,
    "has_verification_request": 16,
    "has_link_shortener": 18,
}


def extract_message_features(message: str):
    """Returns (feature_dict, cleaned_text_for_ml)."""
    message = clean_text(message)
    text_lower = message.lower()
    urls = extract_urls(message)

    feats = {
        "message_length": len(message),
        "num_urgency_words": count_keyword_hits(text_lower, URGENCY_WORDS),
        "num_prize_words": count_keyword_hits(text_lower, PRIZE_WORDS),
        "has_verification_request": int(count_keyword_hits(text_lower, VERIFICATION_WORDS) > 0),
        "has_payment_request": int(count_keyword_hits(text_lower, PAYMENT_WORDS) > 0),
        "has_otp_request": int(count_keyword_hits(text_lower, OTP_WORDS) > 0),
        "has_password_request": int(count_keyword_hits(text_lower, PASSWORD_WORDS) > 0),
        "num_links": len(urls),
        "has_link_shortener": int(has_shortener(urls)),
        "caps_ratio": caps_ratio(message),
        "excessive_punctuation": excessive_punctuation_count(message),
        "exclamation_count": exclamation_count(message),
    }
    return feats, message


def compute_message_feature_risk_score(feats: dict) -> float:
    """Transparent, additive heuristic risk score (0-100) from message features."""
    score = 0.0
    for key, weight in FEATURE_RISK_WEIGHTS.items():
        if feats.get(key):
            score += weight

    score += min(20, feats.get("num_urgency_words", 0) * 8)
    score += min(20, feats.get("num_prize_words", 0) * 8)

    n_links = feats.get("num_links", 0)
    if n_links >= 2:
        score += 14
    elif n_links >= 1:
        score += 8

    if feats.get("caps_ratio", 0) > 0.35:
        score += 8
    if feats.get("excessive_punctuation", 0) >= 2:
        score += 6
    if feats.get("exclamation_count", 0) >= 3:
        score += 5

    return round(min(100.0, score), 2)


def explain_message_features(feats: dict) -> list:
    reasons = []

    if feats["has_otp_request"]:
        reasons.append({"level": "high", "text": "Requests an OTP / verification code"})
    if feats["has_password_request"]:
        reasons.append({"level": "high", "text": "Requests a password"})
    if feats["has_payment_request"]:
        reasons.append({"level": "high", "text": "Requests a payment or bank transfer"})
    if feats["has_verification_request"]:
        reasons.append({"level": "medium", "text": "Asks you to verify your account or identity"})
    if feats["num_prize_words"] >= 1:
        reasons.append({"level": "high", "text": "Claims you've won a prize or reward"})
    if feats["has_link_shortener"]:
        reasons.append({"level": "high", "text": "Contains a shortened URL (hides real destination)"})
    if feats["num_links"] >= 2:
        reasons.append({"level": "medium", "text": "Contains multiple links"})
    elif feats["num_links"] >= 1:
        reasons.append({"level": "low", "text": "Contains a link"})
    if feats["num_urgency_words"] >= 2:
        reasons.append({"level": "high", "text": "Multiple urgency-related phrases detected"})
    elif feats["num_urgency_words"] == 1:
        reasons.append({"level": "medium", "text": "Urgency-related language detected"})
    if feats["caps_ratio"] > 0.35:
        reasons.append({"level": "medium", "text": "Excessive use of capital letters"})
    if feats["excessive_punctuation"] >= 2:
        reasons.append({"level": "low", "text": "Excessive punctuation (e.g. '!!!', '??')"})
    if feats["message_length"] < 15:
        reasons.append({"level": "low", "text": "Very short message (limited context available)"})

    if not reasons:
        reasons.append({"level": "safe", "text": "No major risk indicators found in the message"})

    return reasons
