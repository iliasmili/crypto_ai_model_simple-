import os
import pandas as pd
import joblib

from features import create_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "btc_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Run 'python src/data_loader.py' first."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run 'python src/train_model.py' first."
        )

    data = pd.read_csv(DATA_PATH)
    data = create_features(data)

    latest = data.tail(1)[["ma_5", "ma_10", "Volume"]]
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(latest)

    if prediction[0] == 1:
        print("Price likely to go UP")
    else:
        print("Price likely to go DOWN")

if __name__ == "__main__":
    main()
