import modal
import os
import io

app = modal.App("problem2-autogluon")

# AutoGluon requires specific dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1", "build-essential")
    .pip_install(
        "torch",
        "autogluon.timeseries",
        "pandas",
        "numpy"
    )
)

@app.function(gpu="A10G", image=image, timeout=7200) # 2 hour timeout limit just in case
def train_and_predict(train_csv_bytes: bytes, test_csv_bytes: bytes):
    import logging
    import pandas as pd
    import numpy as np
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Decoding CSVs...")
    train_df = pd.read_csv(io.BytesIO(train_csv_bytes))
    test_df = pd.read_csv(io.BytesIO(test_csv_bytes))
    
    logging.info("Pre-processing dates...")
    def process_dates(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour'] = df['Hour Range'].str.split(' - ').str[0]
        df['timestamp'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Hour'] + ':00')
        df['item_id'] = 'demand'
        return df

    train_df = process_dates(train_df)
    test_df = process_dates(test_df)
    
    # Ensure there are no missing dates in the training sequence by resampling
    logging.info("Filling missing timestamps in training data...")
    train_df = train_df.set_index('timestamp')
    full_idx = pd.date_range(start=train_df.index.min(), end=train_df.index.max(), freq='h')
    train_df = train_df.reindex(full_idx)
    train_df['item_id'] = 'demand'
    train_df['Demand_MW'] = train_df['Demand_MW'].interpolate(method='linear')
    train_df = train_df.reset_index().rename(columns={'index': 'timestamp'})
    
    # Optional: Keep all data (AutoGluon is smart enough to handle anomalies and seasonality)
    # We will use the full 5 years so AutoGluon has maximum statistical power.
    
    logging.info("Converting to AutoGluon TimeSeriesDataFrame...")
    ag_train = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    last_train_date = train_df['timestamp'].max()
    max_test_date = test_df['timestamp'].max()
    horizon = int((max_test_date - last_train_date).total_seconds() / 3600)
    
    logging.info(f"Forecast Horizon: {horizon} hours")
    
    logging.info("Initializing AutoGluon TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=horizon,
        path="ag_models",
        target="Demand_MW",
        eval_metric="MAE"
    )
    
    logging.info("Launching AutoGluon brute-force search (high_quality preset)...")
    # We give it a generous time limit to ensure it trains DeepAR, PatchTST, and tabular models
    predictor.fit(
        ag_train,
        presets="high_quality",
        time_limit=1800 # 30 minutes max
    )
    
    logging.info("AutoGluon training complete. Generating final ensemble predictions...")
    predictions = predictor.predict(ag_train)
    
    # AutoGluon returns a dataframe indexed by ['item_id', 'timestamp']
    predictions = predictions.reset_index()
    
    logging.info("Merging predictions with test set...")
    # Merge back to test_df to ensure exact alignment
    result_df = pd.merge(test_df, predictions, on=['item_id', 'timestamp'], how='left')
    
    # The default prediction column from AutoGluon is 'mean'
    result_df['Demand_MW'] = result_df['mean']
    
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
    output_path = os.path.join(base_dir, 'output', 'submission_autogluon.csv')
    
    print(f"Reading local datasets from {base_dir}...")
    with open(train_path, 'rb') as f:
        train_bytes = f.read()
    with open(test_path, 'rb') as f:
        test_bytes = f.read()
        
    print("Uploading data and unleashing AutoGluon on Modal A10G GPU...")
    print("WARNING: AutoGluon is brute-forcing hundreds of hyperparameters. This may take up to 30 minutes!")
    
    result_bytes = train_and_predict.remote(train_bytes, test_bytes)
    
    print(f"Success! Saving AutoGluon's ultimate ensemble to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(result_bytes)
    print("Done!")
