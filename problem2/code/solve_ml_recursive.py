import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def process_dates(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Hour Range'].str.split(' - ').str[0].str.split(':').str[0].astype(int)
    df['timestamp'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Hour'].astype(str) + ':00:00')
    return df

def extract_features(df):
    # Core time features
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['dayofmonth'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical Encoding
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
    output_path = os.path.join(base_dir, 'output', 'submission_ml_recursive.csv')
    
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train = process_dates(train)
    test = process_dates(test)
    
    # Create continuous full index for train to ensure lags are accurate
    full_idx = pd.date_range(start=train['timestamp'].min(), end=train['timestamp'].max(), freq='h')
    train_full = train.set_index('timestamp').reindex(full_idx)
    train_full['Demand_MW'] = train_full['Demand_MW'].interpolate()
    train_full['Hour'] = train_full.index.hour
    train_full = train_full.reset_index().rename(columns={'index': 'timestamp'})
    
    # Extract lag features
    print("Generating Lag Features...")
    train_full['lag_24'] = train_full['Demand_MW'].shift(24)
    train_full['lag_168'] = train_full['Demand_MW'].shift(168)
    train_full['lag_8736'] = train_full['Demand_MW'].shift(8736) # Exactly 52 weeks (364 days)
    
    train_full = extract_features(train_full)
    
    # Drop rows with NaN due to lag shift (this will drop the first year 2019)
    train_model = train_full.dropna(subset=['lag_8736']).copy()
    
    features = [
        'Hour', 'dayofweek', 'dayofmonth', 'month', 'dayofyear', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'lag_24', 'lag_168', 'lag_8736'
    ]
    
    X_train = train_model[features]
    y_train = np.log1p(train_model['Demand_MW'])
    
    print("Training The Holy Trinity (XGBoost, LightGBM, CatBoost)...")
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, subsample=0.8, random_state=42, verbose=0)
    
    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    
    print("Beginning Recursive Forecasting for December 2023...")
    
    # Prepare historical demand array (from the very beginning to Nov 2023)
    history_demand = train_full['Demand_MW'].tolist()
    
    # We will simulate the test set row by row
    test_extracted = extract_features(test)
    predictions = []
    
    for i in range(len(test_extracted)):
        row = test_extracted.iloc[i:i+1].copy()
        
        # Calculate lags from the history array
        # The last element in history_demand is index -1.
        # Demand 24 hours ago is history_demand[-24]
        row['lag_24'] = history_demand[-24]
        row['lag_168'] = history_demand[-168]
        row['lag_8736'] = history_demand[-8736]
        
        X_step = row[features]
        
        p_xgb = np.expm1(xgb.predict(X_step))[0]
        p_lgb = np.expm1(lgb.predict(X_step))[0]
        p_cat = np.expm1(cat.predict(X_step))[0]
        
        step_pred = (p_xgb + p_lgb + p_cat) / 3.0
        
        # Store prediction
        predictions.append(step_pred)
        
        # INJECT PREDICTION INTO HISTORY
        history_demand.append(step_pred)
        
        if (i+1) % 100 == 0:
            print(f"Forecasted {i+1}/744 hours...")
            
    print(f"Saving final predictions to {output_path}...")
    sub = pd.DataFrame({'ID': test['ID'], 'Demand_MW': predictions})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub.to_csv(output_path, index=False)
    print("Done!")
