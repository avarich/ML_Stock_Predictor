import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, add_features, add_target, build_dataset


def test_target_refers_to_next_trading_day(prices):
    df = add_target(prices)
    for i in [0, 10, 100, 198]:
        expected = int(prices["Close"].iloc[i + 1] > prices["Close"].iloc[i])
        assert df["Target"].iloc[i] == expected


def test_last_row_target_unknown_and_dropped(prices):
    assert pd.isna(add_target(prices)["Target"].iloc[-1])
    data = build_dataset(prices)
    assert data["Date"].iloc[-1] == prices["Date"].iloc[-2]


def test_features_calculated_correctly(prices):
    df = add_features(prices)
    i = 30
    close = prices["Close"]
    assert df["Ret_5"].iloc[i] == pytest.approx(close.iloc[i] / close.iloc[i - 5] - 1)
    assert df["Close_over_MA20"].iloc[i] == pytest.approx(
        close.iloc[i] / close.iloc[i - 19 : i + 1].mean()
    )


def test_features_do_not_use_the_future(prices):
    """Changing a later price must not change an earlier day's features."""
    changed = prices.copy()
    changed.loc[150:, "Close"] *= 3
    a, b = add_features(prices), add_features(changed)
    assert a.loc[:149, FEATURE_COLUMNS].equals(b.loc[:149, FEATURE_COLUMNS])


def test_build_dataset_has_no_missing_values(prices):
    data = build_dataset(prices)
    assert not data[FEATURE_COLUMNS + ["Target"]].isna().any().any()
    assert len(data) == len(prices) - 20 - 1  # 20 warm-up rows + unknown last target
