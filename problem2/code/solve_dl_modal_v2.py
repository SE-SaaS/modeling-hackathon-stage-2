import modal
import os
import io
import pandas as pd

app = modal.App("problem2-dl-ensemble-v2")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "neuralforecast",
        "pandas",
        "numpy"
    )
)

@app.function(gpu="A10G", image=image, timeout=3600) # 1 hour timeout
def train_and_predict(train_csv_bytes: bytes, test_csv_bytes: bytes):
    import logging
    import pandas as pd
    import numpy as np
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, PatchTST, LSTM
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Decoding CSVs...")
    train_df = pd.read_csv(io.BytesIO(train_csv_bytes))
    test_df = pd.read_csv(io.BytesIO(test_csv_bytes))
    
    logging.info("Pre-processing dates...")
    def process_dates(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour'] = df['Hour Range'].str.split(' - ').str[0]
        df['ds'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Hour'] + ':00')
        df['unique_id'] = '1'
        return df

    train_df = process_dates(train_df)
    train_df = train_df.rename(columns={'Demand_MW': 'y'})
    
    test_df = process_dates(test_df)
    
    # Ensure there are no missing dates in the training sequence by resampling
    logging.info("Filling missing timestamps in training data...")
    train_df = train_df.set_index('ds')
    full_idx = pd.date_range(start=train_df.index.min(), end=train_df.index.max(), freq='h')
    train_df = train_df.reindex(full_idx)
    train_df['unique_id'] = '1'
    train_df['y'] = train_df['y'].interpolate(method='linear')
    train_df = train_df.reset_index().rename(columns={'index': 'ds'})
    
    # GRANDMASTER TRICK 1: Drop COVID Data
    logging.info("Dropping 2019-2021 Data to mitigate Concept Drift...")
    train_df = train_df[train_df['ds'] >= '2022-01-01'].copy()
    
    # We need to forecast for exactly len(test_df) steps, but test_df might have gaps.
    # The safest way is to forecast for the maximum horizon from the end of train, 
    # and then merge with test_df based on 'ds'.
    last_train_date = train_df['ds'].max()
    max_test_date = test_df['ds'].max()
    horizon = int((max_test_date - last_train_date).total_seconds() / 3600)
    
    logging.info(f"Forecast Horizon: {horizon} hours")
    
    logging.info("Initializing Grandmaster Deep Learning Models...")
    # GRANDMASTER TRICK 2: Expand Receptive Field (input_size=720) and add StandardScaler
    models = [
        NHITS(h=horizon, input_size=720, max_steps=1000, scaler_type='standard'),
        PatchTST(h=horizon, input_size=720, max_steps=1000, scaler_type='standard'),
        LSTM(h=horizon, input_size=720, max_steps=1000, scaler_type='standard')
    ]
    
    nf = NeuralForecast(models=models, freq='h')
    
    logging.info("Training Models on A10G Cloud GPU (this will take about 15 minutes)...")
    nf.fit(df=train_df[['unique_id', 'ds', 'y']])
    
    logging.info("Generating forecasts...")
    forecasts = nf.predict()
    
    logging.info("Ensembling model predictions...")
    # Average the predictions of all 3 models
    forecasts['Demand_MW'] = (forecasts['NHITS'] + forecasts['PatchTST'] + forecasts['LSTM']) / 3.0
    
    logging.info("Merging with test set...")
    # Merge back to test_df to ensure we only have the required rows in the exact order
    result_df = pd.merge(test_df, forecasts, on=['unique_id', 'ds'], how='left')
    
    # If there are any NaNs due to mismatch, fill with the last known value or mean
    if result_df['Demand_MW'].isna().any():
        logging.warning("Some dates in test.csv did not match the forecast. Forward filling...")
        result_df['Demand_MW'] = result_df['Demand_MW'].ffill().bfill()
        
    out_df = pd.DataFrame({
        'ID': result_df['ID'],
        'Demand_MW': result_df['Demand_MW']
    })
    
    csv_buffer = io.StringIO()
    out_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

@app.local_entrypoint()
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'datasets', 'train.csv')
    test_path = os.path.join(base_dir, 'datasets', 'test.csv')
    output_path = os.path.join(base_dir, 'output', 'submission_dl_modal_v2.csv')
    
    print(f"Reading local datasets from {base_dir}...")
    with open(train_path, 'rb') as f:
        train_bytes = f.read()
    with open(test_path, 'rb') as f:
        test_bytes = f.read()
        
    print("Uploading data and starting NeuralForecast v2 on Modal GPU...")
    print("This will take a little longer (about 15 mins) due to the massive 720-hour window!")
    
    result_bytes = train_and_predict.remote(train_bytes, test_bytes)
    
    print(f"Success! Saving ensembled predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(result_bytes)
    print("Done!")
