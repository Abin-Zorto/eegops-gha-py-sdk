import argparse
import pandas as pd
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser("Split data for RAI analysis")
    parser.add_argument("--input_mltable", type=str, help="Input MLTable path")
    parser.add_argument("--train_data", type=str, help="Output path for train data")
    parser.add_argument("--test_data", type=str, help="Output path for test data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for splitting")
    return parser.parse_args()

def main():
    mlflow.start_run()
    args = parse_args()
    
    # Read the MLTable
    df = pd.read_parquet(args.input_mltable)
    
    # Remove Participant column
    df = df.drop(['Participant'], axis=1)
    
    # Split the data
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df['Remission']
    )
    
    # Save train and test data as MLTable format
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # Save data files
    train_df.to_parquet(train_path / "data.parquet")
    test_df.to_parquet(test_path / "data.parquet")
    
    # Create MLTable files
    mltable_content = """type: mltable
paths:
  - file: ./data.parquet
transformations:
  - read_parquet: {}"""
    
    with open(train_path / "MLTable", "w") as f:
        f.write(mltable_content)
    with open(test_path / "MLTable", "w") as f:
        f.write(mltable_content)
    
    mlflow.log_metric("train_samples", len(train_df))
    mlflow.log_metric("test_samples", len(test_df))
    mlflow.end_run()

if __name__ == "__main__":
    main() 