import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
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
    
    # Log data information
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Number of unique participants: {df['Participant'].nunique()}")
    
    # Prepare data
    X = df.drop(["Participant", "Remission"], axis=1)
    y = df["Remission"]
    groups = df["Participant"]
    
    # Create base classifier
    logger.info("Training the Decision Tree Classifier...")
    base_clf = DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=20
    )
    
    # Create calibrated classifier with cv=0 just for predict_proba
    logger.info("Creating calibrated classifier...")
    clf = CalibratedClassifierCV(
        base_clf,
        cv=0,  # No CV in CalibratedClassifierCV
        method='sigmoid'
    )
    
    # Fit the model
    logger.info("Fitting model...")
    clf.fit(X, y)

    # Test predictions
    try:
        test_proba = clf.predict_proba(X.iloc[:1])
        logger.info(f"Probability prediction test successful: {test_proba}")
    except Exception as e:
        logger.error(f"Probability prediction test failed: {str(e)}")
        raise

    # Save model
    logger.info(f"Saving model to: {model_output_path}")
    model_wrapper = ModelWrapper(clf)
    signature = mlflow.models.infer_signature(X, clf.predict_proba(X)[:, 1])
    
    mlflow.pyfunc.save_model(
        path=model_output_path,
        python_model=model_wrapper,
        signature=signature
    )
    
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_wrapper,
        registered_model_name=args.model_name
    )
    
    logger.info("Model training and saving completed successfully")

if __name__ == "__main__":
    main()