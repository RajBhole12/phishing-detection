"""
url_features.py
----------------
Pure, side-effect-free URL feature extraction.

IMPORTANT SAFETY NOTE:
This module NEVER opens a network connection, resolves DNS, fetches a page,
or executes anything related to the submitted URL. Every feature is derived
purely from the URL *string* itself (lexical analysis), which is why this
approach is safe to run on arbitrary, potentially malicious user input.
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "account", "banking", "confirm",
    "signin", "sign-in", "password", "credential", "wallet", "webscr",
    "suspend", "unlock", "urgent", "billing", "invoice", "support",
    "authenticate", "recovery", "limited", "click", "free", "bonus",
    "gift", "prize", "winner", "reactivate"
]

SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "rebrand.ly", "cutt.ly", "shorte.st", "tiny.cc",
    "shorturl.at", "rb.gy", "v.gd"
]

BRAND_KEYWORDS = [
    "paypal", "apple", "microsoft", "amazon", "google", "facebook",
    "netflix", "bankofamerica", "wellsfargo", "chase", "instagram",
    "whatsapp", "outlook", "dropbox", "linkedin", "ebay"
]

IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
IPV4_IN_HOST = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    length = len(s)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z]+://", url):
        url = "http://" + url
    return url


def extract_features(raw_url: str) -> dict:
    """
    Extract a fixed-order dictionary of lexical/structural features from a
    URL string. Does not perform any network I/O.
    """
    url = _normalize_url(raw_url)
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Strip port from netloc for domain-only calculations
    domain = hostname

    subdomain_parts = domain.split(".") if domain else []
    # crude heuristic: anything beyond "domain.tld" (or domain.co.uk-like) counts as subdomain levels
    num_subdomains = max(0, len(subdomain_parts) - 2)

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9]", url)),
        "num_subdomains": num_subdomains,
        "num_query_params": len(query.split("&")) if query else 0,
        "num_equal_signs": url.count("="),
        "num_ampersands": url.count("&"),
        "num_at_symbols": url.count("@"),
        "num_percent": url.count("%"),
        "has_ip_address": int(bool(IPV4_IN_HOST.search(domain))),
        "has_at_symbol": int("@" in url),
        "has_https": int(scheme == "https"),
        "has_port": int(":" in netloc and not netloc.endswith(":80") and not netloc.endswith(":443")),
        "is_shortened": int(any(short in domain for short in SHORTENER_DOMAINS)),
        "has_suspicious_keyword": int(any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS)),
        "num_suspicious_keywords": sum(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS),
        "has_brand_keyword_in_subdomain": int(
            any(b in ".".join(subdomain_parts[:-2]) for b in BRAND_KEYWORDS) if len(subdomain_parts) > 2 else False
        ),
        "double_slash_redirect": int("//" in path),
        "has_double_dash": int("--" in domain),
        "digit_ratio": round(sum(c.isdigit() for c in url) / max(len(url), 1), 4),
        "hostname_entropy": _shannon_entropy(domain),
        "path_entropy": _shannon_entropy(path),
        "tld_length": len(domain.split(".")[-1]) if "." in domain else 0,
        "starts_with_www": int(domain.startswith("www.")),
        "has_non_standard_tld": int(
            "." in domain and domain.split(".")[-1] not in
            {"com", "org", "net", "edu", "gov", "io", "co", "us", "uk", "info"}
        ),
    }
    return features


FEATURE_ORDER = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_digits", "num_special_chars", "num_subdomains", "num_query_params",
    "num_equal_signs", "num_ampersands", "num_at_symbols", "num_percent",
    "has_ip_address", "has_at_symbol", "has_https", "has_port",
    "is_shortened", "has_suspicious_keyword", "num_suspicious_keywords",
    "has_brand_keyword_in_subdomain", "double_slash_redirect",
    "has_double_dash", "digit_ratio", "hostname_entropy", "path_entropy",
    "tld_length", "starts_with_www", "has_non_standard_tld",
]


def features_to_vector(feature_dict: dict) -> list:
    """Return features in a fixed, model-compatible order."""
    return [feature_dict[k] for k in FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Feature-based risk scoring (used to blend with the ML model's probability
# so the final risk score can genuinely land in the "Suspicious" middle band
# instead of only ever being pushed to the extremes by a binary classifier).
# Weights are additive point values, capped at 100.
# ---------------------------------------------------------------------------
FEATURE_RISK_WEIGHTS = {
    "has_ip_address": 26,
    "has_at_symbol": 16,
    "is_shortened": 20,
    "has_brand_keyword_in_subdomain": 20,
    "has_double_dash": 9,
    "double_slash_redirect": 8,
    "has_non_standard_tld": 7,
    "has_port": 6,
}


def compute_feature_risk_score(feature_dict: dict) -> float:
    """
    Produces a 0-100 heuristic risk score derived purely from extracted URL
    features (independent of the ML model). This is combined with the
    model's phishing probability in app.py to produce the final risk score,
    so a URL with several mild risk indicators -- but not enough for the
    model to be fully confident -- can correctly land in the Suspicious
    band instead of always being forced to 0 or 100.
    """
    score = 0.0

    for key, weight in FEATURE_RISK_WEIGHTS.items():
        if feature_dict.get(key):
            score += weight

    # Graduated contributions (not simple booleans)
    n_kw = feature_dict.get("num_suspicious_keywords", 0)
    score += min(24, n_kw * 9)

    n_sub = feature_dict.get("num_subdomains", 0)
    if n_sub >= 3:
        score += 16
    elif n_sub == 2:
        score += 8

    url_len = feature_dict.get("url_length", 0)
    if url_len > 120:
        score += 14
    elif url_len > 75:
        score += 7

    if feature_dict.get("hostname_entropy", 0) > 3.8:
        score += 9

    if not feature_dict.get("has_https"):
        score += 8

    if feature_dict.get("digit_ratio", 0) > 0.25:
        score += 6

    return round(min(100.0, score), 2)


def explain_features(feature_dict: dict) -> list:
    """
    Produce a list of human-readable risk factors from extracted features.
    Each item: {"level": "high"|"medium"|"low", "text": str}
    """
    reasons = []

    if feature_dict["has_ip_address"]:
        reasons.append({"level": "high", "text": "IP address used instead of a domain name"})
    if feature_dict["has_at_symbol"]:
        reasons.append({"level": "high", "text": "'@' symbol found in URL (can hide the real destination)"})
    if feature_dict["is_shortened"]:
        reasons.append({"level": "high", "text": "URL shortening service detected"})
    if feature_dict["num_subdomains"] >= 3:
        reasons.append({"level": "high", "text": "Excessive number of subdomains"})
    elif feature_dict["num_subdomains"] == 2:
        reasons.append({"level": "medium", "text": "Multiple subdomains detected"})
    if feature_dict["has_brand_keyword_in_subdomain"]:
        reasons.append({"level": "high", "text": "Brand name used in a subdomain (possible impersonation)"})
    if feature_dict["num_suspicious_keywords"] >= 2:
        reasons.append({"level": "high", "text": "Multiple suspicious keywords detected (e.g. 'login', 'verify', 'secure')"})
    elif feature_dict["num_suspicious_keywords"] == 1:
        reasons.append({"level": "medium", "text": "Suspicious keyword detected in URL"})
    if feature_dict["url_length"] > 75:
        reasons.append({"level": "medium", "text": "Unusually long URL"})
    if feature_dict["has_double_dash"]:
        reasons.append({"level": "medium", "text": "Double hyphen in domain (often used for typosquatting)"})
    if feature_dict["double_slash_redirect"]:
        reasons.append({"level": "medium", "text": "Double slash found in URL path (possible redirect trick)"})
    if feature_dict["has_non_standard_tld"]:
        reasons.append({"level": "low", "text": "Uncommon top-level domain"})
    if feature_dict["hostname_entropy"] > 3.8:
        reasons.append({"level": "medium", "text": "High randomness in domain name"})

    if feature_dict["has_https"]:
        reasons.append({"level": "safe", "text": "HTTPS (encrypted connection) detected"})
    else:
        reasons.append({"level": "medium", "text": "No HTTPS detected (unencrypted connection)"})
    if feature_dict["starts_with_www"] and not feature_dict["has_ip_address"]:
        reasons.append({"level": "safe", "text": "Standard 'www' domain structure"})
    if not reasons:
        reasons.append({"level": "safe", "text": "No major risk indicators found"})

    return reasons
