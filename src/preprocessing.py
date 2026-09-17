"""
preprocessing.py
-----------------
Loads the raw URL dataset, runs it through the feature extractor, cleans it,
and produces train/test splits ready for model training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.url_features import extract_features, FEATURE_ORDER

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "phishing.csv")


class DatasetNotFoundError(Exception):
    pass


def load_raw_dataset():
    if not os.path.exists(DATA_PATH):
        raise DatasetNotFoundError(
            f"No dataset found at {DATA_PATH}. Add a CSV with columns "
            f"'url,label' (label: 1=phishing, 0=legitimate), or run "
            f"'python src/generate_dataset.py' to create a starter dataset."
        )
    df = pd.read_csv(DATA_PATH)
    if not {"url", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'url' and 'label' columns.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["url"])
    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame):
    """Runs every URL through the feature extractor. Returns (X_df, y)."""
    feature_rows = []
    for url in df["url"]:
        try:
            feats = extract_features(url)
        except Exception:
            feats = {k: 0 for k in FEATURE_ORDER}
        feature_rows.append(feats)
    X = pd.DataFrame(feature_rows)[FEATURE_ORDER]
    y = df["label"].values
    return X, y


def get_train_test_data(test_size=0.2, random_state=42):
    """
    Full pipeline: load -> clean -> feature-extract -> split -> scale.
    Returns dict with everything downstream code needs.
    """
    raw = load_raw_dataset()
    clean = clean_dataset(raw)
    X, y = build_feature_matrix(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_ORDER,
        "dataset_size": len(clean),
        "n_features": len(FEATURE_ORDER),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_distribution": {
            "legitimate": int((y == 0).sum()),
            "phishing": int((y == 1).sum()),
        },
    }
