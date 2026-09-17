"""
text_common_model.py
----------------------
Generic TF-IDF + scikit-learn text classification trainer/predictor, shared
by the NEW Email and Message phishing detectors (src/email_detection/ and
src/message_detection/). Each of those modules is a thin wrapper that just
points this shared logic at its own model directory and dataset -- this
avoids duplicating the same training/prediction code twice while keeping
the two detectors' *data* and *saved models* completely separate.

Per the project requirements, models and vectorizers here are saved with
Python's built-in `pickle` module -- NOT joblib (joblib is only used by the
existing, untouched Website/URL pipeline in src/train_models.py).

This module has no relationship to, and does not import from or modify,
the existing URL detection pipeline.
"""

import os
import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def get_model_definitions():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": MultinomialNB(),
        # LinearSVC has no predict_proba; CalibratedClassifierCV adds one via
        # Platt scaling so it can participate in the same risk-blend layer.
        "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=3),
    }


def train_and_save(dataset_path: str, model_dir: str, text_col: str = "text",
                    label_col: str = "label", verbose: bool = True):
    """
    Loads a labeled (text,label) CSV, trains a TF-IDF vectorizer + several
    classifiers, evaluates them on a held-out split, saves everything with
    pickle, and records which model performed best (by ROC-AUC).
    """
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"No dataset found at {dataset_path}. Provide a CSV with columns "
            f"'{text_col},{label_col}' ({label_col}: 1=phishing, 0=legitimate)."
        )

    df = pd.read_csv(dataset_path).dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42,
        stratify=df[label_col]
    )

    vectorizer = TfidfVectorizer(
        max_features=4000, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = {}
    trained = {}

    for name, model in get_model_definitions().items():
        if verbose:
            print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba") else y_pred.astype(float))

        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        trained[name] = model

    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])

    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    for name, model in trained.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        with open(os.path.join(model_dir, fname), "wb") as f:
            pickle.dump(model, f)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_info = {
        "best_model": best_name,
        "metric_used": "roc_auc",
        "dataset_size": len(df),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "class_distribution": {
            "legitimate": int((df[label_col] == 0).sum()),
            "phishing": int((df[label_col] == 1).sum()),
        },
    }
    with open(os.path.join(model_dir, "best_model.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    if verbose:
        print(f"Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")

    return results, best_info


def models_are_trained(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "best_model.json")) and \
        os.path.exists(os.path.join(model_dir, "vectorizer.pkl"))


def get_best_model_name(model_dir: str):
    path = os.path.join(model_dir, "best_model.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("best_model")


def load_vectorizer(model_dir: str):
    path = os.path.join(model_dir, "vectorizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Vectorizer not found -- train the model first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(model_dir: str, name: str):
    fname = name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(model_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file for '{name}' not found -- train the model first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba_text(model_dir: str, text: str):
    """Returns (phishing_probability, model_name_used)."""
    if not models_are_trained(model_dir):
        raise FileNotFoundError(
            "This model has not been trained yet. Run the corresponding "
            "train_*.py script first."
        )
    best_name = get_best_model_name(model_dir)
    vectorizer = load_vectorizer(model_dir)
    model = load_model(model_dir, best_name)
    X = vectorizer.transform([text])
    prob = float(model.predict_proba(X)[0][1])
    return prob, best_name


def get_metrics(model_dir: str):
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
