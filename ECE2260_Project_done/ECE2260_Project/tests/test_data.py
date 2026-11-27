import pandas as pd
import pytest

from src.data import load_stock_data, validate_stock_data


def test_loader_parses_dollars_sorts_and_renames(nasdaq_csv):
    df = load_stock_data(nasdaq_csv)
    assert df["Date"].is_monotonic_increasing
    assert pd.api.types.is_float_dtype(df["Close"])
    assert df["Close"].gt(0).all()


def test_loader_drops_duplicate_dates(tmp_path, nasdaq_csv):
    text = nasdaq_csv.read_text().splitlines()
    dup = tmp_path / "dup.csv"
    dup.write_text("\n".join(text + [text[1]]) + "\n")
    df = load_stock_data(dup)
    assert df["Date"].is_unique
    assert df.attrs["duplicates_dropped"] == 1


def test_missing_close_raises(prices):
    prices.loc[5, "Close"] = float("nan")
    with pytest.raises(ValueError, match="Missing"):
        validate_stock_data(prices)


def test_non_positive_price_raises(prices):
    prices.loc[5, "Close"] = 0
    with pytest.raises(ValueError, match="negative closing"):
        validate_stock_data(prices)


def test_too_little_history_raises(prices):
    with pytest.raises(ValueError, match="Only 10 rows"):
        validate_stock_data(prices.head(10))


def test_long_gap_is_reported_not_changed(prices):
    gapped = prices.drop(index=range(50, 60)).reset_index(drop=True)
    report = validate_stock_data(gapped)
    assert len(report["long_gaps"]) == 1
    assert len(gapped) == len(prices) - 10  # nothing was filled in
