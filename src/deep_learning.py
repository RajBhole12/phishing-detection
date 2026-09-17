"""
deep_learning.py
-----------------
Defines and trains a small feed-forward Artificial Neural Network (ANN)
using TensorFlow/Keras for phishing URL classification.

This module is written against the real TensorFlow/Keras API. If
TensorFlow is not installed in the current environment, functions here
raise a clear ImportError so the rest of the app (which treats the DL
model as optional) can fall back to the best classical ML model instead
of crashing or faking results.

Install TensorFlow with:  pip install tensorflow
"""

import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "ann_model.keras")


def is_tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def build_ann(input_dim: int):
    """Builds a compact, regularized ANN suitable for tabular URL features."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_ann(X_train_scaled, y_train, X_test_scaled, y_test, epochs=40, batch_size=32, verbose=0):
    """Trains the ANN and returns (model, history, metrics_dict)."""
    if not is_tensorflow_available():
        raise ImportError(
            "TensorFlow is not installed in this environment. "
            "Run 'pip install tensorflow' to enable the deep learning model."
        )
    from tensorflow import keras
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    model = build_ann(X_train_scaled.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    y_prob = model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(DL_MODEL_PATH)

    return model, history, metrics


def load_ann():
    if not is_tensorflow_available():
        raise ImportError("TensorFlow is not installed; cannot load ANN model.")
    if not os.path.exists(DL_MODEL_PATH):
        raise FileNotFoundError("ANN model has not been trained yet.")
    from tensorflow import keras
    return keras.models.load_model(DL_MODEL_PATH)


def predict_ann(model, X_scaled_row):
    """X_scaled_row: 2D array of shape (1, n_features)."""
    prob = float(model.predict(np.array(X_scaled_row), verbose=0).ravel()[0])
    return prob
