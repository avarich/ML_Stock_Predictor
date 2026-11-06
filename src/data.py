import pandas as pd


def load_stock_data():

    filepath = "data/raw/NVDA.csv"

    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df = df.drop_duplicates(subset="Date")

    df = df.reset_index(drop=True)

    return df