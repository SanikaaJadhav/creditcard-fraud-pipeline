# src/ingest.py
import pandas as pd
from sqlalchemy import create_engine, text
import os

DB_PATH = "data/fraud.db"
CSV_PATH = "data/creditcard.csv"

def load_and_ingest():
    print("📥 Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Feature engineering before storing
    df['hour_of_day'] = (df['Time'] // 3600 % 24).astype(int)
    df['amount_log'] = df['Amount'].apply(lambda x: __import__('numpy').log1p(x))
    df['is_high_value'] = (df['Amount'] > 200).astype(int)
    df['transaction_id'] = range(1, len(df) + 1)

    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    # Write to SQLite
    print("🗄️  Writing to SQLite...")
    engine = create_engine(f"sqlite:///{DB_PATH}")

    df.to_sql("transactions", engine, if_exists="replace", index=False)

    # Create useful views for analytics
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE VIEW IF NOT EXISTS fraud_summary AS
            SELECT
                hour_of_day,
                COUNT(*) as total_transactions,
                SUM(Class) as fraud_count,
                ROUND(AVG(Class) * 100, 4) as fraud_rate_pct,
                ROUND(AVG(Amount), 2) as avg_amount,
                ROUND(MAX(Amount), 2) as max_amount
            FROM transactions
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """))
        conn.commit()

    print(f"✅ SQLite DB created at {DB_PATH}")
    print(f"   Table: transactions | Rows: {len(df):,}")
    return df, engine

if __name__ == "__main__":
    df, engine = load_and_ingest()