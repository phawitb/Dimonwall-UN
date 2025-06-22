# inference.py

import pandas as pd
import joblib
import os

# --- Configuration ---
input_file = 'Mockdata_Clean.csv'
model_path = os.path.join('output', 'best_model.pkl')
output_file = os.path.join('output', 'predicted_results.csv')

feature_cols = [
    'R-test Score', 'PT Score', 'Year of Service', 'Deployment Service Year',
    'KPI from Supervisor', 'KPI from Peers', 'KPI from Subordinate',
    'Active Duty Day in One Year', 'Number of Mission Assigned',
    'Number of Mission Succeed', 'UN English Test Score',
    'UN Knowledge Test Score', 'Timeliness_normalize',
    'Deployment_experience_score', 'Language_skill_score',
    'Rate_of_success_100_scale'
]

# --- Load model ---
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit()

model = joblib.load(model_path)
print(f"✅ Loaded best model from: {model_path}")

# --- Load data ---
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

# --- Predict ---
X = df[feature_cols]
df['Predicted_UNMEM_Score'] = model.predict(X).round(2)

# --- Sort by prediction (high → low) ---
df_sorted = df.sort_values(by='Predicted_UNMEM_Score', ascending=False)

# --- Save result ---
df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"📄 Prediction results saved to: {output_file}")

# --- Show top 5 ---
preview_cols = ['Name', 'Predicted_UNMEM_Score']
print("\n🔝 Top 5 predictions:")
print(df_sorted[preview_cols].head())
