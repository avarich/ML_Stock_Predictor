import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """Synthetic random-walk prices in the same shape load_stock_data returns."""
    rng = np.random.default_rng(0)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2023-01-02", periods=n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        }
    )


@pytest.fixture
def nasdaq_csv(tmp_path, prices):
    """Same text format as data/raw/NVDA.csv (newest first, $ prices, Close/Last)."""
    df = pd.DataFrame(
        {
            "Date": prices["Date"].dt.strftime("%m/%d/%Y"),
            "Close/Last": prices["Close"].map(lambda v: f"${v:.3f}"),
            "Volume": prices["Volume"],
            "Open": prices["Close"].map(lambda v: f"${v:.3f}"),
            "High": prices["Close"].map(lambda v: f"${v:.3f}"),
            "Low": prices["Close"].map(lambda v: f"${v:.3f}"),
        }
    ).iloc[::-1]
    path = tmp_path / "TEST.csv"
    df.to_csv(path, index=False)
    return path
