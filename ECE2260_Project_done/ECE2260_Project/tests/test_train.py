import pytest

from src.features import FEATURE_COLUMNS, build_dataset
from src.train import build_models, chronological_split, get_xy, train_model


def test_split_is_chronological_with_no_overlap(prices):
    data = build_dataset(prices)
    train, test = chronological_split(data.sample(frac=1, random_state=1), 0.8)
    assert train["Date"].max() < test["Date"].min()
    assert len(train) + len(test) == len(data)
    assert len(train) == int(len(data) * 0.8)


def test_split_rejects_bad_fraction(prices):
    with pytest.raises(ValueError):
        chronological_split(build_dataset(prices), 1.0)


def test_inputs_exclude_target_and_date(prices):
    X, y = get_xy(build_dataset(prices))
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}


def test_all_models_train_and_predict(prices):
    train, test = chronological_split(build_dataset(prices))
    X_train, y_train = get_xy(train)
    X_test, _ = get_xy(test)
    for model in build_models().values():
        preds = train_model(model, X_train, y_train).predict(X_test)
        assert len(preds) == len(test)
        assert set(preds) <= {0, 1}
