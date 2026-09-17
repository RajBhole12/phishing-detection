# PhishGuard AI — AI-Based Phishing Website Detection System

A full-stack cybersecurity web application that analyzes URLs with trained
Machine Learning and Deep Learning models and returns a real, explainable
prediction (Legitimate / Suspicious / Phishing) with a risk score,
confidence, model attribution, and feature-level reasoning.

## Quick Start

```bash
pip install -r requirements.txt

# Optional — only needed for the neural-network model, safe to skip:
pip install -r requirements-tensorflow.txt

# 1. Provide a dataset (see "Dataset" below), then:
python src/generate_dataset.py   # only runs if data/phishing.csv doesn't already exist
python src/train_models.py       # trains all models, saves them to /models
python src/evaluate_models.py    # generates evaluation plots into static/images

# 2. Run the app
python app.py
```

Then open **http://localhost:5000**.

## Troubleshooting installation

**"No module named flask" even after installing packages manually:**
This almost always means `pip install -r requirements.txt` failed partway
through and silently aborted — most commonly because of the TensorFlow line.
Modern `pip` resolves every package in the file *before* installing any of
them, so if one line can't be satisfied (wrong Python version, unsupported
OS/architecture, no matching wheel), **none** of the packages get installed,
even ones with no problem of their own like Flask.

That's why TensorFlow now lives in its own optional file
(`requirements-tensorflow.txt`) instead of the main `requirements.txt` — the
app is built to work fully without it (see "Deep Learning Model" below).

If you still see errors:
1. Check your Python version: `python --version` (3.9–3.12 is safest; very
   new versions like 3.13 may not yet have wheels for some packages).
2. Make sure you're installing into the same Python/environment you're
   running the app with — `pip install` and `python app.py` must use the
   same interpreter. If you have multiple Python installs, try
   `python -m pip install -r requirements.txt` and `python -m flask` /
   `python app.py` with the same `python` command each time.
3. Consider using a virtual environment to avoid conflicts with other
   projects:
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Upgrade pip first: `python -m pip install --upgrade pip`.
5. Install packages one at a time to see exactly which one fails:
   `pip install Flask numpy pandas scikit-learn joblib matplotlib seaborn`.

## Dataset

This project expects `data/phishing.csv` with two columns: `url,label`
(`label`: `1` = phishing, `0` = legitimate). Real-world datasets you can drop
in as-is include Kaggle's "Phishing Website Detector" dataset, PhishTank
exports, or the UCI Phishing Websites dataset (reshaped to `url,label`).

**If no dataset is present**, `src/generate_dataset.py` builds a
structurally-realistic starter dataset (~2,400 URLs) by composing real
legitimate domains with known phishing URL construction patterns (IP-hosted
links, brand names stuffed into subdomains, URL shorteners, typosquatting,
suspicious keywords). Labels are assigned by construction, and every model
metric you see in the app (accuracy, precision, recall, F1, ROC-AUC,
confusion matrices) is computed for real from an actual train/test split —
nothing is hard-coded. For production use, swap in a real dataset for
stronger real-world generalization.

## Deep Learning Model

The ANN (`src/deep_learning.py`) is a genuine TensorFlow/Keras Sequential
model (Dense + BatchNorm + Dropout layers) trained with early stopping. If
TensorFlow isn't installed, the app automatically skips ANN training and
selection so the six classical ML models still work — nothing fails silently
or fakes a result.

## How Predictions Work

1. The submitted URL string is parsed and 31 lexical/structural features are
   extracted (`src/features/url_features.py`) — length, entropy, subdomain
   count, IP-address hosting, suspicious keywords, etc.
2. **The target site is never opened, fetched, or executed** — this is
   string-level analysis only, which is what makes it safe to run on
   arbitrary/malicious input.
3. The best model (selected automatically by ROC-AUC during training) scores
   the feature vector.
4. The probability is mapped to a label + 0–100 risk score, and a
   human-readable explanation is generated from the actual extracted
   features.

## Project Structure

```
phishing-detection/
├── app.py                     # Flask app + API routes
├── requirements.txt
├── data/phishing.csv          # dataset (user-supplied or generated)
├── models/                    # trained models, scaler, metrics (generated)
├── database/
│   ├── database.py            # SQLite persistence layer
│   └── scans.db               # generated on first run
├── src/
│   ├── features/url_features.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── deep_learning.py
│   └── generate_dataset.py
├── templates/                 # Jinja2 pages
└── static/{css,js,images}/
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | `{ "url": "..." }` → prediction, risk score, confidence, features, reasons |
| `/api/history` | GET / DELETE | list / clear scan history (supports `search`, `prediction`, `sort_by`, `sort_dir`) |
| `/api/statistics` | GET | dashboard totals, recent scans, activity over time |
| `/api/model-performance` | GET | metrics for every trained model + best-model info |
| `/api/train` | POST | kicks off a background training run |
| `/api/train-status` | GET | polling endpoint for training progress |

## Notes

- If models haven't been trained, the scanner clearly tells the user to
  train them rather than faking a prediction.
- If the dataset is missing, training clearly reports that instead of
  generating fake results.
- All database queries are parameterized (no SQL injection surface).

## New: Email & Message Phishing Detection

Two additional, fully independent detectors were added alongside the
existing Website/URL scanner — nothing about the URL pipeline changed.

- **Email Scanner** (`/email-scanner`) — paste a sender address, subject,
  and body. Analyzed by `src/email_detection/` using its own TF-IDF +
  scikit-learn text classifier (Logistic Regression / Naive Bayes / Linear
  SVM, best one auto-selected by ROC-AUC) plus a feature-based heuristic
  layer (OTP/password/payment requests, urgency language, link count,
  sender domain checks, etc).
- **Message Scanner** (`/message-scanner`) — paste an SMS/chat message.
  Same architecture as the email detector via `src/message_detection/`,
  tuned for short-form messages (prize claims, account-verification asks,
  urgency, links).

Both detectors:
- Use their **own separate datasets** (`data/emails.csv`, `data/messages.csv`,
  auto-generated as structurally-realistic starter data if missing) and
  **their own models**, stored with Python's built-in `pickle` (not
  joblib) in `models/email/` and `models/message/`.
- Share a generic training/prediction module (`src/text_common_model.py`)
  and a generic three-level risk-blending module (`src/risk_blend.py`) —
  both new files, independent of the URL scanner's own blending logic in
  `app.py`.
- Are binary text classifiers, same as the URL model. "Suspicious" is a
  risk-tier applied on top of the model output, not a third training class.
- Never access a real mailbox, SMS inbox, or contacts — only analyze text
  pasted in manually.

Train them with:
```bash
python src/email_detection/train_email_model.py
python src/message_detection/train_message_model.py
```
(Pre-trained models ship with this project, so this step is optional
unless you want to retrain on your own dataset.)

### Dashboard & History extensions

The dashboard now also shows separate Emails/Messages Analyzed, Phishing,
Suspicious, and Legitimate counts — the existing Website/URL stat cards and
chart are computed exactly as before (scoped to `scan_type='Website'`,
which all pre-existing scan records are automatically migrated to).

Scan History now has a `scan_type` column (Website / Email / Message) with
a type filter. Existing history records and search/sort/clear behavior are
unchanged.
