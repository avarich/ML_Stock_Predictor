import pandas as pd

# Longest rolling window used in features.py is 20 days, plus 1 row lost to
# pct_change and 1 row lost to the shifted target. We want far more than that
# so there is real data left to train and test on.
MIN_ROWS = 100

# Gaps longer than this many calendar days between consecutive rows are
# reported (a normal long weekend / holiday is at most 4 days).
MAX_NORMAL_GAP_DAYS = 4


def load_stock_data(filepath="data/raw/NVDA.csv"):

    # read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # match the column name the feature code expects
    df = df.rename(columns={"Close/Last": "Close"})

    # convert the "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # prices may be text like "$182.31": strip symbols and convert to numbers
    for col in ["Close", "Open", "High", "Low"]:
        if col not in df.columns:
            # only Date, Close, and Volume are required
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.replace("$", "", regex=False)
            df[col] = cleaned.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="raise")

    # make sure volume is numeric
    df["Volume"] = pd.to_numeric(df["Volume"], errors="raise")

    # sort the DataFrame by the "Date" column in ascending order
    df = df.sort_values("Date")

    # count duplicate dates so the run can report them
    n_duplicates = int(df.duplicated(subset="Date").sum())
    # drop any duplicate rows based on the "Date" column
    df = df.drop_duplicates(subset="Date")

    # reset the index of the DataFrame and drop the old index
    df = df.reset_index(drop=True)
    df.attrs["duplicates_dropped"] = n_duplicates

    # raise on bad data instead of silently fixing it
    validate_stock_data(df)

    return df


def validate_stock_data(df, min_rows=MIN_ROWS):
    """Check the loaded data and return a report dict.

    Raises ValueError for problems that would make the experiment invalid
    (missing values, non-positive prices, too little history). Calendar gaps
    are only reported, because market holidays create normal gaps.
    Nothing is filled in or changed here.
    """
    required = ["Date", "Close", "Volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df[["Date", "Close", "Volume"]].isna().any().any():
        raise ValueError("Missing values found in Date, Close or Volume")
    if (df["Close"] <= 0).any():
        raise ValueError("Zero or negative closing prices found")
    if (df["Volume"] < 0).any():
        raise ValueError("Negative volume found")
    if not df["Date"].is_monotonic_increasing or df["Date"].duplicated().any():
        raise ValueError("Dates must be sorted ascending with no duplicates")
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} rows of data; need at least {min_rows} to build "
            "20-day rolling features and still have rows left to train and test"
        )

    gaps = df["Date"].diff().dt.days
    long_gaps = df.loc[gaps > MAX_NORMAL_GAP_DAYS, "Date"].dt.date.tolist()

    return {
        "rows": len(df),
        "start": df["Date"].iloc[0],
        "end": df["Date"].iloc[-1],
        "duplicates_dropped": df.attrs.get("duplicates_dropped", 0),
        "long_gaps": long_gaps,
    }
