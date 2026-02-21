"""
Train the regime classifier (Random Forest) on labeled historical data.
Saves model + feature names to model/regime_model.pkl.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, MODEL_DIR, HISTORICAL_MONTHS
from data.fetch_data import fetch_historical_bars
from model.label import label_historical_days, build_training_dataset
from features.engineer import FEATURE_NAMES


def train(save_path: str = None, months: int = None) -> dict:
    """
    Fetch data, label days, build features, train RF, save model.
    Returns dict with accuracy, cv score, and class distribution.
    """
    save_path = save_path or MODEL_PATH
    months = months or HISTORICAL_MONTHS
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Fetching historical bars...")
    ohlcv = fetch_historical_bars(months=months)
    if ohlcv.empty or len(ohlcv) < 500:
        raise ValueError("Insufficient historical data. Need at least ~500 bars.")

    print("Labeling days...")
    labeled = label_historical_days(ohlcv)
    if labeled.empty or labeled["label"].nunique() < 2:
        raise ValueError("Could not label enough days or only one class.")

    print("Building feature matrix...")
    train_df = build_training_dataset(ohlcv, labeled)
    if train_df.empty or len(train_df) < 15:
        raise ValueError("Training set too small (need at least 15 labeled days).")

    X = train_df[FEATURE_NAMES].copy()
    y = train_df["label"].copy()

    # Fill any remaining NaNs
    X = X.fillna(X.median())
    y = y.dropna()
    X = X.loc[y.index]
    if len(X) < 10:
        raise ValueError("Too few samples after dropping NaN labels.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, min_samples_leaf=3)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(y) // 5), scoring="accuracy")
    clf.fit(X_scaled, y)
    train_acc = (clf.predict(X_scaled) == y).mean()

    payload = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "classes": list(clf.classes_),
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    return {
        "train_accuracy": train_acc,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_samples": len(y),
        "class_counts": y.value_counts().to_dict(),
    }


if __name__ == "__main__":
    try:
        metrics = train()
        print("Training done.")
        print("Train accuracy:", round(metrics["train_accuracy"], 4))
        print("CV accuracy:   ", round(metrics["cv_accuracy_mean"], 4), "+/-", round(metrics["cv_accuracy_std"], 4))
        print("Samples:       ", metrics["n_samples"])
        print("Class counts:  ", metrics["class_counts"])
    except Exception as e:
        print("Error:", e)
        raise
