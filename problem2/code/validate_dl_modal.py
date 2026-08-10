import modal
import os
import io
import pandas as pd

app = modal.App("problem2-validate")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "neuralforecast", "pandas", "numpy", "scikit-learn")
)

@app.function(gpu="A10G", image=image, timeout=3600)
def validate(train_csv_bytes: bytes):
    import pandas as pd
    import numpy as np
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from sklearn.metrics import mean_absolute_error
    
    train_df = pd.read_csv(io.BytesIO(train_csv_bytes))
    
    def process_dates(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour'] = df['Hour Range'].str.split(' - ').str[0]
        df['ds'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Hour'] + ':00')
        df['unique_id'] = '1'
        return df

    train_df = process_dates(train_df)
    train_df = train_df.rename(columns={'Demand_MW': 'y'})
    
    train_df = train_df.set_index('ds')
    full_idx = pd.date_range(start=train_df.index.min(), end=train_df.index.max(), freq='h')
    train_df = train_df.reindex(full_idx)
    train_df['unique_id'] = '1'
    train_df['y'] = train_df['y'].interpolate(method='linear')
    train_df = train_df.reset_index().rename(columns={'index': 'ds'})
    
    # Filter COVID
    train_df = train_df[train_df['ds'] >= '2022-01-01'].copy()
    
    val_mask = train_df['ds'] >= '2023-11-01'
    train_data = train_df[~val_mask]
    val_data = train_df[val_mask]
    
    horizon = len(val_data)
    
    # Train NHITS with larger input size and scaling
    models = [
        NHITS(h=horizon, input_size=720, max_steps=1000, scaler_type='standard')
    ]
    
    nf = NeuralForecast(models=models, freq='h')
    nf.fit(df=train_data[['unique_id', 'ds', 'y']])
    
    forecasts = nf.predict()
    
    val_data = val_data.merge(forecasts, on=['unique_id', 'ds'], how='inner')
    mae = mean_absolute_error(val_data['y'], val_data['NHITS'])
    
    return float(mae)

@app.local_entrypoint()
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    
    with open(train_path, 'rb') as f:
        train_bytes = f.read()
        
    mae = validate.remote(train_bytes)
    print(f"VAL_DL_MAE: {mae}")
