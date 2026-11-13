import pandas as pd


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add Target = 1 if the next trading day's close is higher than today's."""
    df = df.copy()
    next_close = df["Close"].shift(-1)
    df["Target"] = (next_close > df["Close"]).astype(int)
    # The last row has no next day, so its target is unknown.
    df.loc[next_close.isna(), "Target"] = pd.NA
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features that use only the current day and earlier days."""
    df = df.copy()
    close, volume = df["Close"], df["Volume"]

    # Daily return, used as the base for volatility
    daily_ret = close.pct_change()
    df["Ret_1"] = daily_ret

    # Returns over longer horizons
    df["Ret_5"] = close.pct_change(5)
    df["Ret_10"] = close.pct_change(10)

    # Moving averages as ratios to the current close
    df["Close_over_MA5"] = close / close.rolling(5).mean()
    df["Close_over_MA20"] = close / close.rolling(20).mean()

    # Rolling volatility of daily returns
    df["Vol_5"] = daily_ret.rolling(5).std()
    df["Vol_20"] = daily_ret.rolling(20).std()

    # Volume and volume relative to its rolling average
    df["Volume"] = volume
    df["Volume_over_MA20"] = volume / volume.rolling(20).mean()

    return df


FEATURE_COLUMNS = [
    "Ret_1", "Ret_5", "Ret_10",
    "Close_over_MA5", "Close_over_MA20",
    "Vol_5", "Vol_20",
    "Volume", "Volume_over_MA20",
]


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create target and features, then drop rows with missing values."""
    df = add_target(df)
    df = add_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    df["Target"] = df["Target"].astype(int)
    return df
