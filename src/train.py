# src/train.py
import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
from src.preprocess import preprocess

def train():
    X_train, X_test, y_train, y_test, feature_names = preprocess()

    print("\n🚀 Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Model Performance:")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print(f"ROC-AUC Score:          {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC Score:           {average_precision_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (correct legit):  {tn:,}")
    print(f"  False Positives (false alarms):   {fp:,}")
    print(f"  False Negatives (missed fraud):   {fn:,}")
    print(f"  True Positives  (caught fraud):   {tp:,}")
    print(f"\n  Fraud caught: {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")

    os.makedirs("exports", exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — Fraud Detection')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exports/pr_curve.png", dpi=150)
    plt.close()
    print("📈 Saved: exports/pr_curve.png")

    importance = pd.Series(model.feature_importances_, index=feature_names)
    top15 = importance.nlargest(15)
    plt.figure(figsize=(8, 6))
    top15.sort_values().plot(kind='barh', color='steelblue')
    plt.title('Top 15 Feature Importances — XGBoost')
    plt.tight_layout()
    plt.savefig("exports/feature_importance.png", dpi=150)
    plt.close()
    print("📊 Saved: exports/feature_importance.png")

    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("\n✅ Model saved to models/xgb_model.pkl")

    return model, X_test, y_test, y_prob, feature_names

if __name__ == "__main__":
    train()