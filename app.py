"""
app.py
------
Flask application entrypoint for the AI-Based Phishing Website Detection
System. Serves the frontend pages and the JSON API used by the frontend
JavaScript.

Run:
    python app.py

Make sure models are trained first:
    python src/train_models.py
"""

import os
import re
import json
import math
import threading

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

from src.features.url_features import (
    extract_features, features_to_vector, explain_features, FEATURE_ORDER,
    compute_feature_risk_score,
)
from src import deep_learning as dl
from src.train_models import run_training_pipeline
from src.evaluate_models import generate_all_plots
from src.preprocessing import DatasetNotFoundError
from database import database as db

# --- NEW: Email & Message phishing detection (fully independent pipelines,
# does not import from or modify anything in the URL detection path above) ---
from src.email_detection import email_features as ef
from src.email_detection import email_model as em
from src.message_detection import message_features as mf
from src.message_detection import message_model as mm
from src.risk_blend import blend_scores as blend_text_scores

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
BEST_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")

app = Flask(__name__)
db.init_db()

training_lock = threading.Lock()
training_status = {"in_progress": False, "message": "Idle", "error": None}

URL_INPUT_RE = re.compile(r"^\S{3,2048}$")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def models_are_trained() -> bool:
    return os.path.exists(BEST_MODEL_INFO_PATH) and os.path.exists(METRICS_PATH)


def get_best_model_name():
    if not os.path.exists(BEST_MODEL_INFO_PATH):
        return None
    with open(BEST_MODEL_INFO_PATH) as f:
        return json.load(f).get("best_model")


def load_prediction_model(name: str):
    """Loads either a joblib sklearn model or the Keras ANN by name."""
    if name == "Artificial Neural Network":
        return dl.load_ann(), "ann"
    fname = name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file for '{name}' not found. Please train models first.")
    return joblib.load(path), "sklearn"


def get_scaler():
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Scaler not found. Please train models first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Three-level risk classification layer
# ---------------------------------------------------------------------------
# NOTE: The underlying ML models (Logistic Regression, Random Forest, etc.)
# are trained as BINARY classifiers (legitimate=0 / phishing=1) -- that has
# not changed, and this app does not claim otherwise. A binary classifier's
# predicted probability tends to cluster near 0.0 or 1.0, which means a
# naive "0.4-0.7 = Suspicious" cutoff on the raw probability almost never
# fires in practice.
#
# To get a genuinely explainable three-level outcome, the final risk score
# blends two independently computed signals:
#   1. The trained model's phishing probability (the actual ML prediction)
#   2. A transparent, feature-based heuristic risk score (see
#      compute_feature_risk_score in src/features/url_features.py), which
#      accumulates points for concrete signals like IP-hosted URLs, brand
#      names in a subdomain, suspicious keywords, excessive subdomains, etc.
#
# The blend lets a URL that the model is very confident about, but that
# still carries a couple of mild risk indicators (or vice versa), land in
# the Suspicious band -- driven by real, inspectable features rather than
# a random or hard-coded assignment.
MODEL_WEIGHT = 0.6
FEATURE_WEIGHT = 0.4
CALIBRATION_TEMPERATURE = 3.2  # >1 softens an overconfident classifier's probability

RISK_LOW_HIGH = 35   # 0-35 -> Legitimate
RISK_MID_HIGH = 70   # 36-70 -> Suspicious, 71-100 -> Phishing


def calibrate_probability(p: float, temperature: float = CALIBRATION_TEMPERATURE) -> float:
    """
    Temperature-scales a model's predicted probability. Models trained on
    cleanly-separable data (including our synthetic starter dataset) tend
    to be overconfident, pushing nearly every probability to 0.0 or 1.0.
    Temperature scaling is a standard, well-established calibration
    technique (see Guo et al., 2017 "On Calibration of Modern Neural
    Networks") that pulls a logit toward the decision boundary before
    re-applying the sigmoid, producing a smoother, more realistic
    probability distribution without changing which class is favored.
    """
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    logit = math.log(p / (1 - p))
    calibrated_logit = logit / temperature
    return 1 / (1 + math.exp(-calibrated_logit))


def compute_final_assessment(prob_phishing: float, feats: dict):
    """
    Combines the model's (calibrated) phishing probability with a
    feature-based risk score to produce the final label, 0-100 risk score,
    and confidence. Deterministic: the same URL (same features + same
    model) always produces the same result -- nothing here is randomized.
    """
    calibrated_prob = calibrate_probability(prob_phishing)
    model_score = calibrated_prob * 100.0
    feature_score = compute_feature_risk_score(feats)

    risk_score = (MODEL_WEIGHT * model_score) + (FEATURE_WEIGHT * feature_score)
    risk_score = round(max(0.0, min(100.0, risk_score)), 1)

    if risk_score <= RISK_LOW_HIGH:
        label = "Legitimate"
    elif risk_score <= RISK_MID_HIGH:
        label = "Suspicious"
    else:
        label = "Phishing"

    confidence = compute_confidence(risk_score)

    return {
        "label": label,
        "risk_score": risk_score,
        "confidence": confidence,
        "model_probability": round(model_score, 2),
        "feature_risk_score": round(feature_score, 2),
    }


def compute_confidence(risk_score: float) -> float:
    """
    Confidence reflects how decisively the blended risk score sits within
    its own band, rather than how far it is from a single midpoint. A score
    sitting near the middle of the Suspicious band is inherently more
    ambiguous (lower confidence) than one sitting at the extreme ends of
    the Legitimate or Phishing bands.
    """
    if risk_score <= RISK_LOW_HIGH:
        dist, span = (RISK_LOW_HIGH - risk_score), RISK_LOW_HIGH
    elif risk_score <= RISK_MID_HIGH:
        band_center_dist = min(risk_score - RISK_LOW_HIGH, RISK_MID_HIGH - risk_score)
        dist, span = band_center_dist, (RISK_MID_HIGH - RISK_LOW_HIGH) / 2
    else:
        dist, span = (risk_score - RISK_MID_HIGH), (100 - RISK_MID_HIGH)

    ratio = dist / span if span else 0
    confidence = 50 + (ratio * 48)
    return round(min(99.0, max(50.0, confidence)), 1)


def build_explanation(label: str, risk_score: float, model_probability: float,
                       feature_risk_score: float, model_name: str, reasons: list) -> str:
    n_flags = sum(1 for r in reasons if r["level"] in ("high", "medium"))
    return (
        f"The {model_name} model estimated a {model_probability:.1f}% phishing probability. "
        f"Combined with a feature-based risk score of {feature_risk_score:.1f}/100 from "
        f"{n_flags} notable URL risk indicator(s), the final blended risk score is "
        f"{risk_score}/100, which falls in the {label.upper()} range."
    )


def is_valid_url_input(raw_url: str) -> bool:
    if not raw_url or not isinstance(raw_url, str):
        return False
    raw_url = raw_url.strip()
    if len(raw_url) < 3 or len(raw_url) > 2048:
        return False
    if not URL_INPUT_RE.match(raw_url):
        return False
    return True


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", models_trained=models_are_trained())


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", models_trained=models_are_trained())


@app.route("/scanner")
def scanner():
    return render_template("scanner.html", models_trained=models_are_trained(),
                            best_model=get_best_model_name())


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/models")
def models_page():
    return render_template("models.html", models_trained=models_are_trained())


@app.route("/training")
def training_page():
    dataset_exists = os.path.exists(os.path.join(APP_DIR, "data", "phishing.csv"))
    return render_template("training.html", models_trained=models_are_trained(),
                            dataset_exists=dataset_exists,
                            tensorflow_available=dl.is_tensorflow_available())


# --- NEW: Email & Message scanner pages (additive; existing routes above are untouched) ---

@app.route("/email-scanner")
def email_scanner_page():
    return render_template("email_scanner.html", models_trained=em.models_are_trained())


@app.route("/message-scanner")
def message_scanner_page():
    return render_template("message_scanner.html", models_trained=mm.models_are_trained())


# ---------------------------------------------------------------------------
# API: predict
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    raw_url = (payload.get("url") or "").strip()

    if not is_valid_url_input(raw_url):
        return jsonify({"error": "Please provide a valid, non-empty URL (max 2048 characters, no spaces)."}), 400

    if not models_are_trained():
        return jsonify({
            "error": "Models have not been trained yet. Go to the Training page and click "
                     "'Train Models' (or run: python src/train_models.py)."
        }), 503

    best_name = get_best_model_name()

    try:
        feats = extract_features(raw_url)
        vector = pd.DataFrame([features_to_vector(feats)], columns=FEATURE_ORDER)
        scaler = get_scaler()
        vector_scaled = scaler.transform(vector)

        model, kind = load_prediction_model(best_name)
        if kind == "ann":
            prob_phishing = dl.predict_ann(model, vector_scaled)
        else:
            if best_name in ("KNN", "SVM", "Logistic Regression"):
                prob_phishing = float(model.predict_proba(vector_scaled)[0][1])
            else:
                prob_phishing = float(model.predict_proba(vector)[0][1])

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    reasons = explain_features(feats)
    assessment = compute_final_assessment(prob_phishing, feats)
    explanation = build_explanation(
        assessment["label"], assessment["risk_score"], assessment["model_probability"],
        assessment["feature_risk_score"], best_name, reasons,
    )

    db.insert_scan(raw_url, assessment["label"], assessment["risk_score"],
                    assessment["confidence"], best_name)

    return jsonify({
        "url": raw_url,
        "prediction": assessment["label"],
        "risk_score": assessment["risk_score"],
        "confidence": assessment["confidence"],
        "model_used": best_name,
        # Calibrated model probability -- the actual signal used in the blend above.
        "model_probability": assessment["model_probability"],
        "feature_risk_score": assessment["feature_risk_score"],
        "probability_phishing": assessment["model_probability"],
        "probability_legitimate": round(100 - assessment["model_probability"], 2),
        # Raw, uncalibrated model output, kept for transparency only.
        "raw_model_probability": round(prob_phishing * 100, 2),
        "features": feats,
        "risk_factors": reasons,
        "explanation": explanation,
    })


# ---------------------------------------------------------------------------
# API: predict-email / predict-message  (NEW, additive)
# ---------------------------------------------------------------------------
# These use a completely separate NLP + ML pipeline (src/email_detection,
# src/message_detection, src/text_common_model.py) and a separate
# risk-blending module (src/risk_blend.py) from the Website/URL scanner
# above. Nothing here is shared with or modifies the URL prediction path.

MAX_SENDER_LEN = 320
MAX_SUBJECT_LEN = 998
MAX_BODY_LEN = 20000
MAX_MESSAGE_LEN = 5000


@app.route("/api/predict-email", methods=["POST"])
def api_predict_email():
    payload = request.get_json(silent=True) or {}
    sender = (payload.get("sender") or "").strip()
    subject = (payload.get("subject") or "").strip()
    body = (payload.get("body") or "").strip()

    if not body or len(body) < 3:
        return jsonify({"error": "Please paste the email body to analyze."}), 400
    if len(sender) > MAX_SENDER_LEN or len(subject) > MAX_SUBJECT_LEN or len(body) > MAX_BODY_LEN:
        return jsonify({"error": "One or more fields exceed the maximum allowed length."}), 400

    if not em.models_are_trained():
        return jsonify({
            "error": "Email model has not been trained yet. Run: "
                     "python src/email_detection/train_email_model.py"
        }), 503

    try:
        feats, full_text = ef.extract_email_features(sender, subject, body)
        prob_phishing, model_name = em.predict_proba_text(full_text)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    feature_score = ef.compute_email_feature_risk_score(feats)
    assessment = blend_text_scores(prob_phishing, feature_score)
    reasons = ef.explain_email_features(feats)
    explanation = (
        f"The {model_name} text model estimated a {assessment['model_probability']:.1f}% "
        f"phishing probability from the email content. Combined with a feature-based risk "
        f"score of {assessment['feature_risk_score']:.1f}/100, the final blended risk score "
        f"is {assessment['risk_score']}/100, which falls in the {assessment['label'].upper()} range."
    )

    # Store the subject line (not the full email body) as the history identifier.
    identifier = subject[:200] if subject else "(no subject)"
    db.insert_scan(identifier, assessment["label"], assessment["risk_score"],
                    assessment["confidence"], model_name, scan_type="Email",
                    risk_factors=reasons)

    return jsonify({
        "prediction": assessment["label"],
        "risk_score": assessment["risk_score"],
        "confidence": assessment["confidence"],
        "model_used": model_name,
        "model_probability": assessment["model_probability"],
        "feature_risk_score": assessment["feature_risk_score"],
        "features": feats,
        "risk_factors": reasons,
        "explanation": explanation,
    })


@app.route("/api/predict-message", methods=["POST"])
def api_predict_message():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()

    if not message or len(message) < 3:
        return jsonify({"error": "Please paste a message to analyze."}), 400
    if len(message) > MAX_MESSAGE_LEN:
        return jsonify({"error": f"Message exceeds the maximum allowed length ({MAX_MESSAGE_LEN} characters)."}), 400

    if not mm.models_are_trained():
        return jsonify({
            "error": "Message model has not been trained yet. Run: "
                     "python src/message_detection/train_message_model.py"
        }), 503

    try:
        feats, clean_text = mf.extract_message_features(message)
        prob_phishing, model_name = mm.predict_proba_text(clean_text)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    feature_score = mf.compute_message_feature_risk_score(feats)
    assessment = blend_text_scores(prob_phishing, feature_score)
    reasons = mf.explain_message_features(feats)
    explanation = (
        f"The {model_name} text model estimated a {assessment['model_probability']:.1f}% "
        f"phishing probability from the message content. Combined with a feature-based risk "
        f"score of {assessment['feature_risk_score']:.1f}/100, the final blended risk score "
        f"is {assessment['risk_score']}/100, which falls in the {assessment['label'].upper()} range."
    )

    # Store a short, truncated snippet (not the full message) as the history identifier.
    identifier = message[:80] + ("…" if len(message) > 80 else "")
    db.insert_scan(identifier, assessment["label"], assessment["risk_score"],
                    assessment["confidence"], model_name, scan_type="Message",
                    risk_factors=reasons)

    return jsonify({
        "prediction": assessment["label"],
        "risk_score": assessment["risk_score"],
        "confidence": assessment["confidence"],
        "model_used": model_name,
        "model_probability": assessment["model_probability"],
        "feature_risk_score": assessment["feature_risk_score"],
        "features": feats,
        "risk_factors": reasons,
        "explanation": explanation,
    })


# ---------------------------------------------------------------------------
# API: history
# ---------------------------------------------------------------------------

@app.route("/api/history", methods=["GET"])
def api_history():
    search = request.args.get("search")
    prediction_filter = request.args.get("prediction")
    scan_type_filter = request.args.get("type")  # NEW: optional Website/Email/Message filter
    sort_by = request.args.get("sort_by", "scanned_at")
    sort_dir = request.args.get("sort_dir", "desc")
    rows = db.get_history(search=search, prediction_filter=prediction_filter,
                           sort_by=sort_by, sort_dir=sort_dir,
                           scan_type_filter=scan_type_filter)
    return jsonify({"scans": rows, "count": len(rows)})


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    db.clear_history()
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# API: statistics
# ---------------------------------------------------------------------------

@app.route("/api/statistics", methods=["GET"])
def api_statistics():
    stats = db.get_statistics()
    stats["models_trained"] = models_are_trained()
    stats["best_model"] = get_best_model_name()
    return jsonify(stats)


# ---------------------------------------------------------------------------
# API: model performance
# ---------------------------------------------------------------------------

@app.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    if not models_are_trained():
        return jsonify({"error": "Models have not been trained yet."}), 503

    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(BEST_MODEL_INFO_PATH) as f:
        best_info = json.load(f)

    feature_importance = []
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH) as f:
            feature_importance = json.load(f)

    return jsonify({
        "metrics": metrics,
        "best_model_info": best_info,
        "feature_importance": feature_importance,
    })


# ---------------------------------------------------------------------------
# API: train
# ---------------------------------------------------------------------------

def _background_train():
    global training_status
    try:
        training_status = {"in_progress": True, "message": "Training in progress...", "error": None}
        run_training_pipeline(verbose=False)
        generate_all_plots()
        training_status = {"in_progress": False, "message": "Training complete.", "error": None}
    except DatasetNotFoundError as e:
        training_status = {"in_progress": False, "message": "Failed", "error": str(e)}
    except Exception as e:
        training_status = {"in_progress": False, "message": "Failed", "error": str(e)}


@app.route("/api/train", methods=["POST"])
def api_train():
    if training_lock.locked():
        return jsonify({"status": "already_running", "message": "Training already in progress."}), 409

    dataset_path = os.path.join(APP_DIR, "data", "phishing.csv")
    if not os.path.exists(dataset_path):
        return jsonify({
            "error": "No dataset found at data/phishing.csv. Add a real phishing-URL dataset "
                     "(columns: url,label) or run 'python src/generate_dataset.py' to create a starter dataset."
        }), 400

    def run():
        with training_lock:
            _background_train()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return jsonify({"status": "started", "message": "Training started in the background."})


@app.route("/api/train-status", methods=["GET"])
def api_train_status():
    return jsonify(training_status)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
