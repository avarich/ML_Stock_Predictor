import pandas as pd


def load_stock_data(filepath="data/raw/NVDA.csv"):

    df = pd.read_csv(filepath)                #read the CSV file into a DataFrame

    df = df.rename(columns={"Close/Last": "Close"})   #match the column name the feature code expects

    df["Date"] = pd.to_datetime(df["Date"])   #convert the "Date" column to datetime format

    # prices may be stored as text like "$182.31", so strip symbols and convert to numbers
    for col in ["Close", "Open", "High", "Low"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["Volume"] = pd.to_numeric(df["Volume"], errors="raise")   #make sure volume is numeric

    df = df.sort_values("Date")               #sort the DataFrame by the "Date" column in ascending order

    df = df.drop_duplicates(subset="Date")    #drop any duplicate rows based on the "Date" column

    df = df.reset_index(drop=True)            # reset the index of the DataFrame and drop the old index

    # basic validation: report problems instead of silently fixing them
    if df[["Close", "Volume"]].isna().any().any():
        raise ValueError("Missing values found in Close or Volume")
    if (df["Close"] <= 0).any():
        raise ValueError("Zero or negative closing prices found")

    return df