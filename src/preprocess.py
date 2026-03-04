# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
import joblib
import os

DB_PATH = "data/fraud.db"

def preprocess():
    print("📦 Loading data from SQLite...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM transactions", engine)

    # Features: drop non-model columns
    drop_cols = ['transaction_id', 'Class']
    X = df.drop(columns=drop_cols)
    y = df['Class']

    print(f"   Class distribution before SMOTE: {dict(y.value_counts())}")

    # Scale Amount and Time-based features (V1-V28 already PCA scaled)
    scaler = StandardScaler()
    X[['Amount', 'amount_log', 'Time']] = scaler.fit_transform(
        X[['Amount', 'amount_log', 'Time']]
    )

    # Train/test split BEFORE SMOTE — critical to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"   Fraud in test set: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Apply SMOTE only on training data
    print("\n⚖️  Applying SMOTE to training data...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"   Before SMOTE — Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
    print(f"   After SMOTE  — Fraud: {y_train_sm.sum():,} | Legit: {(y_train_sm==0).sum():,}")

    # Save scaler and split data
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump((X_train_sm, X_test, y_train_sm, y_test), "models/data_splits.pkl")

    print("\n✅ Preprocessing complete. Artifacts saved to models/")
    return X_train_sm, X_test, y_train_sm, y_test, X.columns.tolist()

if __name__ == "__main__":
    preprocess()