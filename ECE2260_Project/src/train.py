"""Chronological split, baseline, and models (README steps 5 and 6)."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS

RANDOM_STATE = 42


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    """Earliest rows -> train, latest rows -> test. Never shuffles."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * train_fraction)
    if split_idx == 0 or split_idx == len(df):
        raise ValueError("Split leaves the train or test set empty")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def get_xy(df: pd.DataFrame):
    """Feature matrix and target. Date and Target are never model inputs."""
    X = df[FEATURE_COLUMNS]
    y = df["Target"].astype(int)
    assert "Target" not in X.columns and "Date" not in X.columns
    return X, y


def build_models() -> dict:
    """Baseline plus the models from the README.

    Scaling lives inside the pipeline, so it is fit on the training data only
    and then re-used (not re-fit) when predicting on validation/test data.
    """
    return {
        "baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "random forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
    }


def cross_validate_train(model, X_train, y_train, n_splits: int = 5) -> float:
    """Mean balanced accuracy using time-ordered folds on TRAINING data only.

    This is how models are compared without touching the test period: each
    fold trains on earlier rows and validates on the rows right after them.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="balanced_accuracy"
    )
    return float(scores.mean())


def train_model(model, X_train, y_train):
    """Fit on the training period only and return the fitted model."""
    return model.fit(X_train, y_train)
