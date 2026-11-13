from src.data import load_stock_data
from src.features import build_dataset

raw_df = load_stock_data()
df = build_dataset(raw_df)

print(df.shape)
print(df[["Date", "Close", "Target"]].head())
print(df["Target"].value_counts())

# spot check: the target should be 1 exactly when the next row's Close is higher
i = 100
print(df.iloc[i][["Date", "Close", "Target"]])
print(df.iloc[i + 1][["Date", "Close"]])
