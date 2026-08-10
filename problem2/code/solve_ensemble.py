import pandas as pd
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    
    path_dl = os.path.join(output_dir, 'submission_dl_modal.csv')
    path_ml = os.path.join(output_dir, 'submission_ml_recursive_v2.csv')
    path_ensemble = os.path.join(output_dir, 'submission_final_ensemble.csv')
    
    print("Loading DL submission (score: 6.00)...")
    dl = pd.read_csv(path_dl)
    
    print("Loading ML Recursive v2 submission...")
    ml = pd.read_csv(path_ml)
    
    print("Ensembling (Weighted Average)...")
    # Because DL and ML make completely different errors, an average is very robust.
    # We weight ML Recursive slightly higher because it uses short-term lags (lag_1, lag_2),
    # which makes it incredibly accurate for the first few days of the forecast.
    
    ensemble = dl.copy()
    ensemble['Demand_MW'] = (0.40 * dl['Demand_MW']) + (0.60 * ml['Demand_MW'])
    
    print(f"Saving Final Ensemble to {path_ensemble}...")
    ensemble.to_csv(path_ensemble, index=False)
    print("Done! Ready for submission.")
