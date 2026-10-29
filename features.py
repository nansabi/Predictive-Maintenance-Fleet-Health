import os
import json
import pandas as pd
import numpy as np

# Resolve artifact path directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALING_PARAMS_PATH = os.path.join(BASE_DIR, "scaling_parameters.json")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_cols.json")

# Load preprocessed pipeline parameters once during module load
with open(SCALING_PARAMS_PATH, "r") as f:
    scaling_params = json.load(f)

with open(FEATURE_COLS_PATH, "r") as f:
    feature_cols = json.load(f)

def build_features(readings_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure sorted order
    df = readings_df.sort_values("cycle").reset_index(drop=True).copy()
    
    # Usable sensor list from config
    raw_sensors = [
        "op1", "op2", "s2", "s3", "s4", "s6", "s7", "s8", "s9",
        "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
    ]
    
    # Compute rolling values over 5 and 10 cycle windows
    # Spark's rowsBetween(-(w-1), 0) corresponds to backward rolling including current record
    for w in [5, 10]:
        for s in raw_sensors:
            df[f"{s}_mean{w}"] = df[s].rolling(window=w, min_periods=1).mean()
            # Pandas std uses ddof=1 by default; fill single-item NaN boundary with 0.0
            df[f"{s}_std{w}"] = df[s].rolling(window=w, min_periods=1).std(ddof=1).fillna(0.0)
            
    # First-order difference calculation with 0.0 filling at boundary
    for s in raw_sensors:
        df[f"{s}_diff1"] = df[s].diff(periods=1).fillna(0.0)
        
    # Standardize engineered features using precalculated parameter weights
    for col, stats in scaling_params.items():
        if col in df.columns:
            df[col] = (df[col] - stats["mean"]) / stats["std"]
            
    # Extract only the final row representing the latest cycle state
    final_row = df.tail(1).copy()
    
    # Select and order features to match the champion's column schema strictly
    return final_row[feature_cols].reset_index(drop=True)
