# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
result_csv = os.path.join("output", "model_eval_results.csv")
output_plot_dir = "output"
os.makedirs(output_plot_dir, exist_ok=True)

# --- Load results ---
df = pd.read_csv(result_csv)

# --- Group by model and calculate mean for each metric ---
df_grouped = df.groupby("Model", as_index=False).agg({
    "Train_RMSE": "mean",
    "Test_RMSE": "mean",
    "Train_MAE": "mean",
    "Test_MAE": "mean",
    "Train_R2": "mean",
    "Test_R2": "mean"
})

# --- Helper function: reshape for comparison plot ---
def melt_for_comparison(df_grouped, metric: str):
    return pd.melt(
        df_grouped[["Model", f"Train_{metric}", f"Test_{metric}"]],
        id_vars="Model",
        var_name="Set",
        value_name=metric
    ).replace({f"Train_{metric}": "Train", f"Test_{metric}": "Test"})

# --- Plot function ---
def plot_comparison(metric: str, ylabel: str, color: str):
    data = melt_for_comparison(df_grouped, metric)
    plt.figure()
    sns.barplot(data=data, x="Model", y=metric, hue="Set", palette=color)
    plt.title(f"Average {metric} Comparison (Train vs Test)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    path = os.path.join(output_plot_dir, f"{metric.lower()}_train_vs_test.png")
    plt.savefig(path)
    print(f"✅ Saved: {path}")

# --- Plot all metrics ---
plot_comparison("RMSE", "Root Mean Squared Error", "Blues")
plot_comparison("MAE", "Mean Absolute Error", "Greens")
plot_comparison("R2", "R² Score", "Oranges")
