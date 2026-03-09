import os
import yfinance as yf
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "btc_data.csv")

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = yf.download("BTC-USD", interval="1h", period="60d")

    if df.empty:
        raise ValueError("No data downloaded from Yahoo Finance.")

    # Keep only needed columns
    df = df[['Close', 'Volume']].copy()

    # Flatten columns if MultiIndex exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.to_csv(DATA_PATH, index=False)

    print(f"Data saved to: {DATA_PATH}")
    print(df.tail())

if __name__ == "__main__":
    download_data()
