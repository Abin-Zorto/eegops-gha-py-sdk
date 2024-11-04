import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split
import os
import json

def parse_args():
    parser = argparse.ArgumentParser("Split data for RAI analysis")
    parser.add_argument("--input_mltable", type=str, help="Input MLTable path")
    parser.add_argument("--train_data", type=str, help="Output path for train data")
    parser.add_argument("--test_data", type=str, help="Output path for test data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for splitting")
    return parser.parse_args()

def read_mltable(mltable_path):
    """Read data from an MLTable directory containing a Parquet file."""
    print(f"Checking directory structure for MLTable at {mltable_path}")
    print("Contents of input MLTable directory:")
    for root, dirs, files in os.walk(mltable_path):
        print(root, "contains directories:", dirs, "and files:", files)
    
    parquet_files = list(Path(mltable_path).glob("*.parquet"))
    if not parquet_files:
        print("No Parquet found in root of MLTable directory, checking subdirectories...")
        parquet_files = list(Path(mltable_path).glob("**/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet file found in the MLTable directory: {mltable_path}")
    
    print(f"Loading data from: {parquet_files[0]}")
    return pd.read_parquet(parquet_files[0])

def aggregate_windows_to_patient(df):
    """
    Aggregate window-level features to patient-level features.
    """
    feature_cols = df.columns.difference(['Participant', 'Remission'])
    
    # Define aggregation functions
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    percentiles = [25, 75]
    
    # Create aggregation dictionary for features
    feature_aggs = {col: agg_funcs for col in feature_cols}
    feature_aggs['Remission'] = 'first'
    
    # Aggregate
    patient_df = df.groupby('Participant').agg(feature_aggs)
    
    # Add percentiles
    for col in feature_cols:
        for p in percentiles:
            patient_df[(col, f'percentile_{p}')] = df.groupby('Participant')[col].quantile(p/100)
    
    # Flatten column names
    patient_df.columns = [f"{col}_{agg}" if agg != 'first' else col 
                         for col, agg in patient_df.columns]
    
    # Add number of windows as a feature
    patient_df['n_windows'] = df.groupby('Participant').size()
    
    return patient_df.reset_index()

def main():
    mlflow.start_run()
    args = parse_args()
    
    # Read the MLTable
    df = read_mltable(args.input_mltable)
    print(f"Window-level data loaded - Shape: {df.shape}")
    
    # Aggregate to patient level first
    patient_df = aggregate_windows_to_patient(df)
    print(f"\nAggregated to patient level - Shape: {patient_df.shape}")
    print(f"Number of features per patient: {patient_df.shape[1]-2}")  # -2 for Participant and Remission
    
    # Drop Participant ID before saving (not a feature)
    train_df, test_df = train_test_split(
        patient_df.drop('Participant', axis=1),  # Remove Participant ID
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=patient_df['Remission']
    )
    
    print(f"\nAfter splitting:")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Train set remission patients: {train_df['Remission'].sum()}")
    print(f"Test set remission patients: {test_df['Remission'].sum()}")
    
    # Save train and test data
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    train_df.to_csv(train_path / "data.csv", index=False)
    test_df.to_csv(test_path / "data.csv", index=False)
    
    # Create MLTable definitions
    mltable_content = {
        "type": "mltable",
        "paths": [{"file": "data.csv"}],
        "transformations": [{"read_delimited": {"delimiter": ","}}]
    }
    
    with open(train_path / "MLTable", "w") as f:
        json.dump(mltable_content, f)
    with open(test_path / "MLTable", "w") as f:
        json.dump(mltable_content, f)
    
    # Log metrics
    mlflow.log_metric("total_patients", len(patient_df))
    mlflow.log_metric("train_patients", len(train_df))
    mlflow.log_metric("test_patients", len(test_df))
    mlflow.log_metric("total_features", train_df.shape[1]-1)  # -1 for Remission column
    mlflow.log_metric("train_remission_patients", train_df['Remission'].sum())
    mlflow.log_metric("test_remission_patients", test_df['Remission'].sum())
    
    # Log feature stats
    for column in train_df.columns:
        if column != 'Remission':  # Skip the target variable
            stats = train_df[column].describe()
            for stat in ['mean', 'std', 'min', 'max']:
                mlflow.log_metric(f"train_{column}_{stat}", stats[stat])
    
    print("\nExample features:")
    print(train_df.columns[:10].tolist())
    
    mlflow.end_run()

if __name__ == "__main__":
    main()