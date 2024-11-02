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
        # For classification tasks, return probability of positive class
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(model_input)
            return proba[:, 1]  # Return probabilities for positive class
        return self.model.predict(model_input)

    def predict_proba(self, model_input):
        # Direct method for RAIInsights to call
        return self.model.predict_proba(model_input)

    # This is the main entry point for MLflow models
    def _predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            return self.predict(context, model_input)
        return self.predict(context, pd.DataFrame(model_input))

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
    clf = DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=20  # Helps ensure better probability estimates
    )
    clf.fit(X, y)
    
    # Verify probability predictions work
    try:
        test_proba = clf.predict_proba(X.iloc[:1])
        logger.info(f"Probability prediction test successful: {test_proba}")
    except Exception as e:
        logger.error(f"Probability prediction test failed: {str(e)}")
        raise

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