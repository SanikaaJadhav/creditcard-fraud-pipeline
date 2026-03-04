# src/explain.py
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def explain():
    print("🔍 Loading model and data...")
    model        = joblib.load("models/xgb_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    _, X_test, _, y_test, = joblib.load("models/data_splits.pkl")

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # SHAP expects a sample — full test set is slow, 1000 rows is plenty
    print("⚙️  Computing SHAP values (this takes ~30 seconds)...")
    sample = X_test_df.sample(1000, random_state=42).reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    os.makedirs("exports", exist_ok=True)

    # --- Plot 1: Beeswarm summary ---
    print("📊 Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Feature Impact on Fraud Predictions")
    plt.tight_layout()
    plt.savefig("exports/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: exports/shap_summary.png")

    # --- Plot 2: Mean absolute SHAP (bar) ---
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("Mean SHAP Value — Feature Importance")
    plt.tight_layout()
    plt.savefig("exports/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: exports/shap_bar.png")

    # --- Plot 3: Single transaction explanation ---
    # Find a fraud case in our sample to explain
    y_pred_sample = model.predict(sample)
    fraud_indices = np.where(y_pred_sample == 1)[0].tolist()

    if fraud_indices:
        idx = fraud_indices[0]
        print(f"\n🔎 Explaining individual fraud transaction (sample index {idx})...")
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=sample.iloc[idx],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig("exports/shap_waterfall.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   Saved: exports/shap_waterfall.png")
    else:
        print("   No fraud cases predicted in sample — skipping waterfall plot")

    # --- Top features by mean SHAP ---
    mean_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print("\n🏆 Top 10 features driving fraud predictions:")
    print(mean_shap.head(10).to_string(index=False))
    mean_shap.to_csv("exports/shap_importance.csv", index=False)
    print("\n✅ SHAP analysis complete. All plots saved to exports/")

    return shap_values, sample, feature_names

if __name__ == "__main__":
    explain()