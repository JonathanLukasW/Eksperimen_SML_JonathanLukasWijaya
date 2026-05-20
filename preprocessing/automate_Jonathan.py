"""
automate_Jonathan.py
====================
Script otomatisasi preprocessing dataset CS:GO Match Round Data.

Struktur folder yang diharapkan:
    root/
    ├── dataset_match_cs/
    │   └── dataset.csv          ← input
    └── preprocessing/
        ├── automate_Jonathan.py ← script 
        └── csgo_preprocessed/   ← output otomatis dibuat

"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset_match_cs",
    "dataset.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "csgo_preprocessed"
)

DROP_COLUMNS = [
    "endTScore",      
    "endCTScore",     
    "ctTeam",         
    "tTeam",          
    "roundEndReason", 
    "winningTeam",    
    "losingTeam",     
]

CATEGORICAL_FEATURES = ["mapName", "ctBuyType", "tBuyType"]
TARGET_COLUMN = "winningSide"
UNKNOWN_LABEL = "Unknown"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"ERROR: File tidak ditemukan - {filepath}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(filepath, index_col=0)

    print("INFORMASI DATASET")
    print("=" * 50)
    print(f"Jumlah baris    : {df.shape[0]:,}")
    print(f"Jumlah kolom    : {df.shape[1]}")
    print(f"Kolom           : {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def perform_eda(df: pd.DataFrame) -> None:
    print("\n--- info data ---")
    df.info()

    print("\n--- deskripsi data ---")
    print(df.describe())

    print("\n--- cek missing value ---")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("Tidak ada missing value.")

    print("\n--- distribusi target (winningSide) ---")
    print(df[TARGET_COLUMN].value_counts())


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    Pipeline preprocessing:
    1. buang baris winningSide == 'Unknown'
    2. drop kolom leakage & kurang relevan
    3. pisahkan feature (X) dan target (y)
    4. encoding 
    5. split train/test (80/20, random_state=42)
    6. StandardScaler
    """

    before = len(df)
    df = df[df[TARGET_COLUMN] != UNKNOWN_LABEL].copy()
    print(f"Rows before: {before}  |  Rows after: {len(df)}  |  Dropped: {before - len(df)}")

    df = df.drop(columns=DROP_COLUMNS)
    print("Sisa columns:", list(df.columns))
    print(f"Jumlah kolom sekarang : {df.shape[1]}")

    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    print("Feature matrix shape:", X.shape)
    print("Target vector shape :", y.shape)

    le_features = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X = X.copy()
        X[col] = le.fit_transform(X[col])
        le_features[col] = le
        print(f"  {col}: {list(le.classes_)}")

    le_target = LabelEncoder()
    y = pd.Series(le_target.fit_transform(y), name=TARGET_COLUMN, index=y.index)
    print("Target classes:", le_target.classes_, "-> encoded as", list(range(len(le_target.classes_))))
    print("Target value counts:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    print("Scaler fitted on training data only.")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------------
# Simpan Output
# ---------------------------------------------------------------------------
def save_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_path = os.path.join(output_dir, "train_clean.csv")
    test_path  = os.path.join(output_dir, "test_clean.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved:", train_path, "| shape:", train_df.shape)
    print("Saved:", test_path,  "| shape:", test_df.shape)
    print("\nPreprocessing complete!")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data(DATASET_PATH)
    perform_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    save_artifacts(X_train, X_test, y_train, y_test, OUTPUT_DIR)


if __name__ == "__main__":
    main()