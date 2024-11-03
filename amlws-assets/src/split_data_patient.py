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
    For each feature, calculate mean, std, min, max, and quartiles.
    """
    # Separate features from metadata
    feature_cols = df.columns.difference(['Participant', 'Remission'])
    
    # Define aggregation functions
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    percentiles = [25, 75]  # Add quartiles
    
    # Create aggregation dictionary for features
    feature_aggs = {col: agg_funcs for col in feature_cols}
    
    # Add Remission (take first since it's same for all windows)
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
    
    # Reset index to make Participant a regular column
    patient_df = patient_df.reset_index()
    
    # Add number of windows as a feature
    patient_df['n_windows'] = df.groupby('Participant').size()
    
    return patient_df

def split_patients(df, test_size=0.2, random_state=42):
    """
    Split patient-level data into train and test sets
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Remission']
    )
    
    # Remove Participant column from both sets
    train_df = train_df.drop('Participant', axis=1)
    test_df = test_df.drop('Participant', axis=1)
    
    return train_df, test_df

def main():
    mlflow.start_run()
    args = parse_args()
    
    # Read the MLTable
    df = read_mltable(args.input_mltable)
    print(f"DataFrame loaded - Columns: {df.columns.tolist()}, Shape: {df.shape}")
    
    # Analyze initial patient distribution
    n_patients = df['Participant'].nunique()
    n_remission = df.groupby('Participant')['Remission'].first().sum()
    print(f"\nInitial distribution:")
    print(f"Total patients: {n_patients}")
    print(f"Remission patients: {n_remission}")
    print(f"Non-remission patients: {n_patients - n_remission}")
    
    # Aggregate to patient level
    print("\nAggregating window-level features to patient-level...")
    patient_df = aggregate_windows_to_patient(df)
    print(f"Patient-level features created: {patient_df.shape[1]} features")
    
    # Split the patient-level data
    train_df, test_df = split_patients(
        patient_df,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print(f"\nAfter splitting:")
    print(f"Train set: {len(train_df)} patients")
    print(f"Test set: {len(test_df)} patients")
    print(f"Train remission patients: {train_df['Remission'].sum()}")
    print(f"Test remission patients: {test_df['Remission'].sum()}")
    
    # Save train and test data as CSV
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
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
    mlflow.log_metric("total_patients", n_patients)
    mlflow.log_metric("train_patients", len(train_df))
    mlflow.log_metric("test_patients", len(test_df))
    mlflow.log_metric("train_remission_patients", train_df['Remission'].sum())
    mlflow.log_metric("test_remission_patients", test_df['Remission'].sum())
    mlflow.log_metric("n_features", train_df.shape[1])
    
    # Log some feature statistics
    feature_stats = train_df.describe()
    for column in feature_stats.columns:
        if column != 'Remission':
            for stat in ['mean', 'std', 'min', 'max']:
                mlflow.log_metric(f"train_{column}_{stat}", feature_stats[column][stat])
    
    mlflow.end_run()

if __name__ == "__main__":
    main()