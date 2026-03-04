# src/analytics.py
import pandas as pd
from sqlalchemy import create_engine, text
import os

DB_PATH = "data/fraud.db"

def run_analytics():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    os.makedirs("exports", exist_ok=True)

    queries = {

        "fraud_by_hour": """
            SELECT hour_of_day,
                   COUNT(*) AS total_txns,
                   SUM(Class) AS fraud_count,
                   ROUND(AVG(Class)*100, 4) AS fraud_rate_pct,
                   ROUND(AVG(Amount), 2) AS avg_amount
            FROM transactions
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """,

        "high_value_fraud": """
            SELECT transaction_id,
                   Amount,
                   hour_of_day,
                   Class
            FROM transactions
            WHERE is_high_value = 1
            ORDER BY Amount DESC
            LIMIT 100
        """,

        "fraud_amount_buckets": """
            SELECT
                CASE
                    WHEN Amount < 10   THEN 'Under $10'
                    WHEN Amount < 50   THEN '$10-$50'
                    WHEN Amount < 100  THEN '$50-$100'
                    WHEN Amount < 500  THEN '$100-$500'
                    ELSE 'Over $500'
                END AS amount_bucket,
                COUNT(*) AS total_txns,
                SUM(Class) AS fraud_count,
                ROUND(AVG(Class)*100, 4) AS fraud_rate_pct
            FROM transactions
            GROUP BY amount_bucket
            ORDER BY fraud_count DESC
        """,

        "hourly_fraud_heatmap": """
            SELECT hour_of_day,
                   SUM(Class) AS fraud_count,
                   COUNT(*) AS total,
                   ROUND(SUM(Class)*100.0/COUNT(*), 4) AS fraud_pct
            FROM transactions
            GROUP BY hour_of_day
            ORDER BY fraud_pct DESC
        """,

        "top_fraud_amounts": """
            SELECT transaction_id,
                   Amount,
                   hour_of_day,
                   V1, V2, V3, V4
            FROM transactions
            WHERE Class = 1
            ORDER BY Amount DESC
            LIMIT 50
        """
    }

    print("🔍 Running SQL analytics queries...\n")
    results = {}
    for name, query in queries.items():
        df = pd.read_sql(text(query), engine)
        results[name] = df
        df.to_csv(f"exports/{name}.csv", index=False)
        print(f"✅ {name}: {len(df)} rows → exports/{name}.csv")
        print(df.head(3).to_string())
        print()

    return results

if __name__ == "__main__":
    run_analytics()