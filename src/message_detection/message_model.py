"""
message_model.py
------------------
Thin wrapper around src/text_common_model.py that points the shared TF-IDF
training/prediction logic at the Message detector's own dataset and model
directory. Independent from email_model.py -- separate dataset, separate
trained vectorizer/models, separate pickle files, never mixed with the
existing Website/URL models.
"""

import os

from src import text_common_model as tcm

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATASET_PATH = os.path.join(APP_DIR, "data", "messages.csv")
MODEL_DIR = os.path.join(APP_DIR, "models", "message")


def train_and_save(verbose: bool = True):
    return tcm.train_and_save(DATASET_PATH, MODEL_DIR, verbose=verbose)


def models_are_trained() -> bool:
    return tcm.models_are_trained(MODEL_DIR)


def get_best_model_name():
    return tcm.get_best_model_name(MODEL_DIR)


def predict_proba_text(text: str):
    return tcm.predict_proba_text(MODEL_DIR, text)


def get_metrics():
    return tcm.get_metrics(MODEL_DIR)


def dataset_exists() -> bool:
    return os.path.exists(DATASET_PATH)
