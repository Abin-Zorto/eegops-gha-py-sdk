import argparse
from pathlib import Path
import pandas as pd
import mlflow
import logging
import joblib
from sklearn.tree import DecisionTreeClassifier
from mlflow.pyfunc import PythonModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        # The error suggests model_input is a DataFrame but we're trying to call predict on it
        # Instead, we should use self.model to make predictions
        return self.model.predict(model_input)

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
    X = df.drop(["Participant", "Remission"], axis=1)
    y = df["Remission"]

    # Train the model
    logger.info("Training the Decision Tree Classifier...")
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    logger.info("Model training completed.")

    # Wrap the trained model
    model_wrapper = ModelWrapper(model=clf)

    # Define model signature
    signature = mlflow.models.infer_signature(X, clf.predict(X))

    # Save the MLflow PyFunc model
    logger.info(f"Saving the MLflow PyFunc model to: {model_output_path}")
    mlflow.pyfunc.save_model(
        path=model_output_path,
        python_model=model_wrapper,
        signature=signature
    )

    # Save the trained model using joblib
    joblib.dump(clf, model_output_path / "model.pkl")
    logger.info("Model artifacts saved successfully.")

    # Assuming 'trained_model' is your actual model
    model_wrapper = ModelWrapper(clf)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_wrapper,
        registered_model_name=args.model_name
    )

if __name__ == "__main__":
    main()