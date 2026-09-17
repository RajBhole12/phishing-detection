"""
risk_blend.py
--------------
Shared three-level (Legitimate / Suspicious / Phishing) risk classification
helpers used ONLY by the new Email and Message phishing detectors.

This is a brand-new, independent module. The existing Website/URL scanner's
risk-blending logic already lives inline in app.py (calibrate_probability /
compute_final_assessment / compute_confidence) and is NOT modified or
replaced by this file -- it is duplicated here in generic form so the new
Email/Message features have their own blending layer without touching a
single line of the existing, working URL code path.

Same approach as the URL scanner: a binary text classifier's probability
tends to cluster near 0.0 or 1.0, so we blend the (temperature-calibrated)
model probability with a transparent, feature-based heuristic risk score to
produce a genuinely three-level, explainable result.
"""

import math

RISK_LOW_HIGH = 35   # 0-35 -> Legitimate
RISK_MID_HIGH = 70   # 36-70 -> Suspicious, 71-100 -> Phishing

DEFAULT_MODEL_WEIGHT = 0.6
DEFAULT_FEATURE_WEIGHT = 0.4
DEFAULT_TEMPERATURE = 3.0


def calibrate_probability(p: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """Temperature-scales an (over)confident binary classifier's probability."""
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    logit = math.log(p / (1 - p))
    calibrated_logit = logit / temperature
    return 1 / (1 + math.exp(-calibrated_logit))


def classify_risk_score(risk_score: float, low_high: int = RISK_LOW_HIGH,
                         mid_high: int = RISK_MID_HIGH) -> str:
    if risk_score <= low_high:
        return "Legitimate"
    elif risk_score <= mid_high:
        return "Suspicious"
    return "Phishing"


def compute_confidence(risk_score: float, low_high: int = RISK_LOW_HIGH,
                        mid_high: int = RISK_MID_HIGH) -> float:
    """Confidence reflects how decisively the score sits within its own band."""
    if risk_score <= low_high:
        dist, span = (low_high - risk_score), low_high
    elif risk_score <= mid_high:
        dist, span = min(risk_score - low_high, mid_high - risk_score), (mid_high - low_high) / 2
    else:
        dist, span = (risk_score - mid_high), (100 - mid_high)

    ratio = dist / span if span else 0
    confidence = 50 + (ratio * 48)
    return round(min(99.0, max(50.0, confidence)), 1)


def blend_scores(model_prob: float, feature_score: float,
                  model_weight: float = DEFAULT_MODEL_WEIGHT,
                  feature_weight: float = DEFAULT_FEATURE_WEIGHT,
                  temperature: float = DEFAULT_TEMPERATURE) -> dict:
    """
    Combines a text model's raw phishing probability with a feature-based
    risk score into a final label, 0-100 risk score, and confidence.
    Deterministic -- nothing here is randomized.
    """
    calibrated = calibrate_probability(model_prob, temperature)
    model_score = calibrated * 100.0
    feature_score = max(0.0, min(100.0, feature_score))

    risk_score = (model_weight * model_score) + (feature_weight * feature_score)
    risk_score = round(max(0.0, min(100.0, risk_score)), 1)

    label = classify_risk_score(risk_score)
    confidence = compute_confidence(risk_score)

    return {
        "label": label,
        "risk_score": risk_score,
        "confidence": confidence,
        "model_probability": round(model_score, 2),
        "feature_risk_score": round(feature_score, 2),
    }
