# train.py

import pandas as pd
import numpy as np
import os
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint
import warnings
import random

warnings.filterwarnings('ignore')

# --- Set seed for reproducibility ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- Output folder setup ---
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

eval_result_csv = os.path.join(output_dir, 'model_eval_results.csv')
best_model_path = os.path.join(output_dir, 'best_model.pkl')
input_filename = 'Mockdata_Clean.csv'

# --- Feature and target configuration ---
feature_cols = [
    'R-test Score', 'PT Score', 'Year of Service', 'Deployment Service Year',
    'KPI from Supervisor', 'KPI from Peers', 'KPI from Subordinate',
    'Active Duty Day in One Year', 'Number of Mission Assigned',
    'Number of Mission Succeed', 'UN English Test Score',
    'UN Knowledge Test Score', 'Timeliness_normalize',
    'Deployment_experience_score', 'Language_skill_score',
    'Rate_of_success_100_scale'
]

target_col = 'UNMEM_Mission_success_score'

# --- Load and clean dataset ---
df = pd.read_csv(input_filename)
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED
)

# --- Define models and parameter search space ---
models = {
    'RandomForest': RandomForestRegressor(random_state=SEED),
    'AdaBoost': AdaBoostRegressor(random_state=SEED),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED)
}

param_grids = {
    'RandomForest': {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    },
    'AdaBoost': {
        'n_estimators': randint(50, 300),
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'XGBoost': {
        'n_estimators': randint(100, 300),
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': randint(3, 10),
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.5, 1, 2],
        'gamma': [0, 0.1, 0.3]
    }
}

results = []

best_rmse = float('inf')
best_model = None
best_model_name = None

# --- Evaluation function ---
def evaluate_model(model, X_train, X_test, y_train, y_test):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
        'Train_MAE': mean_absolute_error(y_train, pred_train),
        'Test_MAE': mean_absolute_error(y_test, pred_test),
        'Train_R2': r2_score(y_train, pred_train),
        'Test_R2': r2_score(y_test, pred_test)
    }

# --- Train and evaluate each model ---
for name, model in models.items():
    print(f"\n🔧 Training & tuning: {name}")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[name],
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=SEED,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_est = search.best_estimator_

    metrics = evaluate_model(best_est, X_train, X_test, y_train, y_test)
    metrics['Model'] = name
    metrics['BestParams'] = search.best_params_
    metrics['Seed'] = SEED
    results.append(metrics)

    print(f"✅ {name} - Test RMSE: {metrics['Test_RMSE']:.4f}, R²: {metrics['Test_R2']:.4f}")

    if metrics['Test_RMSE'] < best_rmse:
        best_rmse = metrics['Test_RMSE']
        best_model = best_est
        best_model_name = name

# --- Save results (append mode if exists) ---
df_results = pd.DataFrame(results)

if os.path.exists(eval_result_csv):
    df_existing = pd.read_csv(eval_result_csv)
    df_results = pd.concat([df_existing, df_results], ignore_index=True)

df_results.to_csv(eval_result_csv, index=False)
print(f"\n📄 Saved evaluation metrics to: {eval_result_csv}")

# --- Save best model ---
joblib.dump(best_model, best_model_path)
print(f"🏆 Best model ({best_model_name}) saved to: {best_model_path}")
