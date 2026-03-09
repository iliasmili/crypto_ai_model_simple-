import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Force numeric columns
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["Close", "Volume"]).copy()

    # Features
    df["return"] = df["Close"].pct_change(fill_method=None)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()

    # Target: next close higher than current close
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().copy()
    return df
