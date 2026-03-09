import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from features import create_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "btc_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}. Run 'python src/data_loader.py' first."
        )

    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    X = df[["ma_5", "ma_10", "Volume"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", round(accuracy, 4))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
