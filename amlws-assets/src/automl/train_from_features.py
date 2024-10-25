import argparse
from pathlib import Path
import pandas as pd
import mlflow
import logging
import time
from azureml.core import Run
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature
import json
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str, help="Path to registered features (MLTable)")
    parser.add_argument("--model_output", type=str, help="Path to model output")
    args = parser.parse_args()
    return args

def load_and_validate_data(features_path: Path) -> pd.DataFrame:
    """Load and validate feature data."""
    logger.info(f"Loading features from: {features_path}")
    df = pd.read_parquet(features_path / "features.parquet")
    return df

def train_random_forest(df: pd.DataFrame, groups: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest model with Leave-One-Group-Out cross-validation."""
    X = df.drop(['Participant', 'Remission'], axis=1)
    y = df['Remission']
    logo = LeaveOneGroupOut()
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("auc", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    return model

def save_model(model: RandomForestClassifier, df: pd.DataFrame, output_path: Path):
    """Save the trained model with MLflow."""
    X = df.drop(['Participant', 'Remission'], axis=1)
    signature = infer_signature(X, model.predict(X))
    mlflow.pyfunc.save_model(
        path=str(output_path),
        python_model=model,
        signature=signature
    )

def main():
    args = parse_args()
    mlflow.start_run()
    run = Run.get_context()

    features_path = Path(args.registered_features)
    output_path = Path(args.model_output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df = load_and_validate_data(features_path)
        groups = df['Participant'].values
        model = train_random_forest(df, groups)
        save_model(model, df, output_path)
        logger.info("Model training and saving completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        mlflow.log_metric("training_success", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
