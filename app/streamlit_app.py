# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# ── Load artifacts (cached) ───────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("models/xgb_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

@st.cache_data
def load_analytics():
    engine = create_engine("sqlite:///data/fraud.db")
    fraud_by_hour   = pd.read_sql("SELECT * FROM fraud_summary", engine)
    buckets         = pd.read_csv("exports/fraud_amount_buckets.csv")
    shap_importance = pd.read_csv("exports/shap_importance.csv")
    return fraud_by_hour, buckets, shap_importance

model, scaler, feature_names = load_model()
fraud_by_hour, buckets, shap_importance = load_analytics()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detection System")
st.sidebar.markdown("**UMD Data Science Portfolio Project**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Predict Transaction", "📊 SQL Analytics", "🧠 SHAP Explainability"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💳 Credit Card Fraud Detection Pipeline")
    st.markdown("### End-to-end ML system: XGBoost + SMOTE + SHAP + SQL Analytics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases", "492")
    col3.metric("ROC-AUC Score", "0.9827")
    col4.metric("Fraud Caught", "87.8%")

    st.markdown("---")
    st.markdown("### 📐 Pipeline Architecture")
    st.code("""
    creditcard.csv
         │
         ▼
    SQLite Database  ──►  SQL Analytics (5 business queries)
         │
         ▼
    SMOTE Balancing  (394 → 227,451 fraud samples)
         │
         ▼
    XGBoost Classifier  (300 trees, PR-AUC optimized)
         │
         ▼
    SHAP Explainability  ──►  Per-transaction explanations
         │
         ▼
    Streamlit App  ──►  Real-time fraud scoring
    """, language="text")

    st.markdown("### 🏆 Model Performance")
    perf = pd.DataFrame({
        "Metric": ["ROC-AUC", "PR-AUC", "Precision (Fraud)", "Recall (Fraud)", "F1 (Fraud)"],
        "Score":  [0.9827,    0.8649,   0.58,                0.88,             0.70]
    })
    fig = px.bar(perf, x="Metric", y="Score", color="Score",
                 color_continuous_scale="Blues", range_y=[0, 1])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT TRANSACTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Transaction":
    st.title("🔮 Real-Time Fraud Prediction")
    st.markdown("Adjust transaction features and get an instant fraud probability score.")

    st.markdown("#### Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, max_value=30000.0, value=149.62)
        time   = st.number_input("Time (seconds since first txn)", min_value=0, max_value=172800, value=45000)
    with col2:
        v1  = st.slider("V1",  -30.0, 30.0, -1.36)
        v2  = st.slider("V2",  -30.0, 30.0,  1.19)
        v3  = st.slider("V3",  -30.0, 30.0, -1.36)
        v4  = st.slider("V4",  -10.0, 10.0,  0.88)
    with col3:
        v14 = st.slider("V14", -20.0, 20.0, -0.31)
        v17 = st.slider("V17", -20.0, 20.0, -0.38)
        v12 = st.slider("V12", -20.0, 20.0, -0.18)
        v10 = st.slider("V10", -20.0, 20.0,  0.09)

    if st.button("🔍 Analyze Transaction", type="primary"):
        # Build feature vector matching training schema
        row = {f: 0.0 for f in feature_names}
        row['Amount']      = amount
        row['Time']        = time
        row['amount_log']  = np.log1p(amount)
        row['hour_of_day'] = int((time // 3600) % 24)
        row['is_high_value'] = int(amount > 200)
        row['V1']  = v1;  row['V2']  = v2;  row['V3']  = v3
        row['V4']  = v4;  row['V10'] = v10; row['V12'] = v12
        row['V14'] = v14; row['V17'] = v17

        X_input = pd.DataFrame([row])[feature_names]
        X_input[['Amount', 'amount_log', 'Time']] = scaler.transform(
            X_input[['Amount', 'amount_log', 'Time']]
        )

        prob     = model.predict_proba(X_input)[0][1]
        decision = "🚨 FRAUD" if prob > 0.5 else "✅ LEGITIMATE"
        color    = "red" if prob > 0.5 else "green"

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"## {decision}")
            st.markdown(f"**Fraud Probability: `{prob:.1%}`**")
        with col_b:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={"text": "Fraud Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  40], "color": "#d4edda"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100],"color": "#f8d7da"},
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        # SHAP waterfall for this prediction
        st.markdown("#### 🧠 Why did the model decide this?")
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_input)
        fig2, ax    = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X_input.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 3 — SQL ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📊 SQL Analytics":
    st.title("📊 SQL Analytics Dashboard")
    st.markdown("Business intelligence queries run directly against the SQLite database.")

    tab1, tab2, tab3 = st.tabs(["🕐 Fraud by Hour", "💰 Amount Buckets", "🔎 Raw SQL Query"])

    with tab1:
        st.markdown("#### Fraud Rate by Hour of Day")
        st.caption("Peak fraud occurs at 2 AM — 10x the average rate")
        fig = px.bar(
            fraud_by_hour, x="hour_of_day", y="fraud_rate_pct",
            color="fraud_rate_pct", color_continuous_scale="Reds",
            labels={"hour_of_day": "Hour", "fraud_rate_pct": "Fraud Rate (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fraud_by_hour, use_container_width=True)

    with tab2:
        st.markdown("#### Fraud Count by Transaction Amount")
        fig2 = px.bar(
            buckets, x="amount_bucket", y="fraud_count",
            color="fraud_rate_pct", color_continuous_scale="Oranges",
            labels={"amount_bucket": "Amount Range", "fraud_count": "Fraud Cases"}
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(buckets, use_container_width=True)

    with tab3:
        st.markdown("#### Run your own SQL query")
        default_query = "SELECT hour_of_day, fraud_count, fraud_rate_pct FROM fraud_summary WHERE fraud_rate_pct > 0.5 ORDER BY fraud_rate_pct DESC"
        query = st.text_area("SQL Query", value=default_query, height=100)
        if st.button("▶ Run Query"):
            try:
                engine = create_engine("sqlite:///data/fraud.db")
                result = pd.read_sql(text(query), engine)
                st.success(f"Returned {len(result)} rows")
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Model Explainability")
    st.markdown("Understanding *why* the model flags transactions — critical for compliance and trust.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Importance (SHAP)")
        if os.path.exists("exports/shap_bar.png"):
            st.image("exports/shap_bar.png", use_container_width=True)
    with col2:
        st.markdown("#### SHAP Beeswarm — Impact Distribution")
        if os.path.exists("exports/shap_summary.png"):
            st.image("exports/shap_summary.png", use_container_width=True)

    st.markdown("#### Single Transaction Waterfall")
    st.caption("Each bar shows how much a feature pushed the prediction toward fraud (red) or legit (blue)")
    if os.path.exists("exports/shap_waterfall.png"):
        st.image("exports/shap_waterfall.png", use_container_width=True)

    st.markdown("#### Top Features by Mean |SHAP|")
    fig = px.bar(
        shap_importance.head(15),
        x="mean_abs_shap", y="feature",
        orientation='h', color="mean_abs_shap",
        color_continuous_scale="Blues"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)