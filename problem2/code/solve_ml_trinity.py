import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def preprocess(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Hour Range'].str.split(' - ').str[0].str.split(':').str[0].astype(int)
    
    # Core time features
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofmonth'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([4, 5]).astype(int) # Friday/Saturday weekend usually in some regions, but let's use standard [5, 6] for Sat/Sun
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical Encoding (crucial for time series)
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/23.0)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek']/6.0)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek']/6.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    output_path = os.path.join(base_dir, 'output', 'submission_ml_trinity.csv')
    
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print("Pre-processing and engineering cyclical features...")
    train = preprocess(train)
    test = preprocess(test)
    
    # DROP COVID DATA: Only keep 2022 and 2023 to capture the most recent economic trends
    print("Filtering historical data (Dropping < 2022)...")
    train = train[train['Date'] >= '2022-01-01'].copy()
    
    features = [
        'Hour', 'dayofweek', 'dayofmonth', 'month', 'dayofyear', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    
    X_train = train[features]
    # Log transform the target to stabilize variance
    y_train = np.log1p(train['Demand_MW'])
    
    X_test = test[features]
    
    print("Initializing The Holy Trinity (XGBoost, LightGBM, CatBoost)...")
    # We use conservative hyperparameters to prevent overfitting on the public leaderboard
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    lgb = LGBMRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    cat = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, subsample=0.8, random_state=42, verbose=0)
    
    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    
    print("Training LightGBM...")
    lgb.fit(X_train, y_train)
    
    print("Training CatBoost...")
    cat.fit(X_train, y_train)
    
    print("Generating predictions and reversing log-transform...")
    p_xgb = np.expm1(xgb.predict(X_test))
    p_lgb = np.expm1(lgb.predict(X_test))
    p_cat = np.expm1(cat.predict(X_test))
    
    print("Averaging the ensemble...")
    final_pred = (p_xgb + p_lgb + p_cat) / 3.0
    
    print(f"Saving final predictions to {output_path}...")
    sub = pd.DataFrame({'ID': test['ID'], 'Demand_MW': final_pred})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub.to_csv(output_path, index=False)
    print("Done!")
