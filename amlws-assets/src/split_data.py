import argparse
import pandas as pd
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
    """Read data from an MLTable directory, verifying file structure."""
    # Check and log the directory contents
    print(f"Checking directory structure for MLTable at {mltable_path}")
    print("Contents of input MLTable directory:")
    for root, dirs, files in os.walk(mltable_path):
        print(root, "contains directories:", dirs, "and files:", files)

    # Look for CSV files directly in the directory
    csv_files = list(Path(mltable_path).glob("*.csv"))
    
    # If no CSV found, check subdirectories
    if not csv_files:
        print("No CSV found in root of MLTable directory, checking subdirectories...")
        csv_files = list(Path(mltable_path).glob("**/*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in the MLTable directory: {mltable_path}")
    
    # Load the first CSV file found
    print(f"Loading data from: {csv_files[0]}")
    return pd.read_csv(csv_files[0])

def main():
    mlflow.start_run()
    args = parse_args()
    
    # Read the MLTable
    df = read_mltable(args.input_mltable)
    
    # Confirm DataFrame loaded
    print(f"DataFrame loaded - Columns: {df.columns.tolist()}, Shape: {df.shape}")
    
    # Drop Participant column if it exists
    if 'Participant' in df.columns:
        df = df.drop(['Participant'], axis=1)
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df.get('Remission')  # Use `.get()` to avoid KeyError if missing
    )
    
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
    mlflow.log_metric("train_samples", len(train_df))
    mlflow.log_metric("test_samples", len(test_df))
    mlflow.end_run()

if __name__ == "__main__":
    main()
