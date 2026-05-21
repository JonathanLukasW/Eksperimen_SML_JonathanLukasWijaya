import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

TRAIN_PATH = os.path.join("csgo_preprocessed", "train_clean.csv")
TEST_PATH  = os.path.join("csgo_preprocessed", "test_clean.csv")

print("[INFO] Membaca data hasil preprocessing...")
train_data = pd.read_csv(TRAIN_PATH)
test_data  = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["winningSide"])
y_train = train_data["winningSide"]
X_test  = test_data.drop(columns=["winningSide"])
y_test  = test_data["winningSide"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.autolog()

with mlflow.start_run(run_name="Random_Forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("run berhasil dicatat otomatis oleh Autolog ke MLflow lokal.")