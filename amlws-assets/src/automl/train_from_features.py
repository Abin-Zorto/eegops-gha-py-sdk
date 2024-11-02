import argparse
from pathlib import Path
import pandas as pd
import mlflow
import logging
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneGroupOut
from mlflow.pyfunc import PythonModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        return self.model.predict_proba(model_input)[:, 1]

def parse_args():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str, help="Path to registered features data")
    parser.add_argument("--model_output", type=str, help="Path to model output")
    parser.add_argument("--model_name", type=str, default="eeg_classifier", help="Name under which model will be registered")
    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")
    return args

def main():
    args = parse_args()
    features_path = Path(args.registered_features)
    model_output_path = Path(args.model_output)
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Load training data
    logger.info(f"Loading training data from: {features_path}")
    df = pd.read_parquet(features_path / "features.parquet")
    
    # Extract groups for LOGO CV
    groups = df["Participant"].values
    X = df.drop(["Participant", "Remission"], axis=1)
    y = df["Remission"]

    # Train base model
    logger.info("Training the Decision Tree Classifier...")
    base_clf = DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=20
    )
    
    # Set up LeaveOneGroupOut CV
    logo = LeaveOneGroupOut()
    
    # Wrap with CalibratedClassifierCV using LOGO
    logger.info("Calibrating probabilities using Leave-One-Group-Out cross-validation...")
    clf = CalibratedClassifierCV(
        base_clf, 
        cv=logo, 
        method='sigmoid',
        n_jobs=-1  # Use all available cores
    )
    
    # Fit model with groups
    clf.fit(X, y, groups=groups)

    # Test probability predictions
    try:
        test_proba = clf.predict_proba(X.iloc[:1])
        logger.info(f"Probability prediction test successful: {test_proba}")
        
        # Log number of groups used in calibration
        n_splits = sum(1 for _ in logo.split(X, y, groups))
        logger.info(f"Number of participant groups used in calibration: {n_splits}")
        
    except Exception as e:
        logger.error(f"Probability prediction test failed: {str(e)}")
        raise

    logger.info("Model training completed.")

    # Wrap and save the calibrated model
    model_wrapper = ModelWrapper(model=clf)
    signature = mlflow.models.infer_signature(X, clf.predict_proba(X)[:, 1])
    
    # Save model artifacts
    logger.info(f"Saving the MLflow PyFunc model to: {model_output_path}")
    mlflow.pyfunc.save_model(
        path=model_output_path,
        python_model=model_wrapper,
        signature=signature
    )
    joblib.dump(clf, model_output_path / "model.pkl")
    logger.info("Model artifacts saved successfully.")

    # Register the model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_wrapper,
        registered_model_name=args.model_name
    )

if __name__ == "__main__":
    main()