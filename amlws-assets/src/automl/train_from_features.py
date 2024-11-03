import argparse
from pathlib import Path
import pandas as pd
import mlflow
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str)
    parser.add_argument("--model_output", type=str)
    parser.add_argument("--model_name", type=str, default="automl")
    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    features_path = Path(args.registered_features)
    model_output_path = Path(args.model_output)
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    logger.info(f"Loading training data from: {features_path}")
    df = pd.read_parquet(features_path / "features.parquet")
    
    X = df.drop(["Participant", "Remission"], axis=1)
    y = df["Remission"]
    groups = df["Participant"]
    
    # Initialize model
    logger.info("Training the Random Forest Classifier...")
    clf = RandomForestClassifier(
        random_state=42,
        min_samples_leaf=20
    )
    
    # Train with Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        clf.fit(X_train, y_train)
    
    # Test prediction probability
    test_proba = clf.predict_proba(X.iloc[:1])
    logger.info(f"Probability prediction test successful: {test_proba}")

    # Save model
    logger.info(f"Saving model to: {model_output_path}")
    signature = mlflow.models.infer_signature(X, clf.predict_proba(X)[:, 1])
    
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=model_output_path,
        signature=signature
    )
    
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        registered_model_name=args.model_name
    )
    
    logger.info("Model training and saving completed successfully")

if __name__ == "__main__":
    main()
