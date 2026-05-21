import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    TRAIN_PATH = os.path.join("csgo_preprocessed", "train_clean.csv")
    TEST_PATH  = os.path.join("csgo_preprocessed", "test_clean.csv")

    print("Membaca dataset re-training")
    train_data = pd.read_csv(TRAIN_PATH)
    test_data  = pd.read_csv(TEST_PATH)

    X_train = train_data.drop(columns=["winningSide"])
    y_train = train_data["winningSide"]
    X_test  = test_data.drop(columns=["winningSide"])
    y_test  = test_data["winningSide"]

    mlflow.autolog()

    print(f"Memulai Retraining Model (n_estimators={args.n_estimators}, max_depth={args.max_depth})")
    with mlflow.start_run(run_name="CI_Automated_Retraining"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        print("Proses training berhasil!")