from src.data import load_stock_data

df = load_stock_data()
print(df.dtypes)
print(df[["Date", "Close"]].iloc[[0, -1]])
