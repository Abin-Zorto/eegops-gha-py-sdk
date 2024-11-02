import argparse
from pathlib import Path
import pandas as pd
import mlflow
import logging
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from typing import Dict, Any, Tuple
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WrappedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            return self.model.predict(pd.DataFrame(model_input))
        return self.model.predict(model_input)
    
    def predict_proba(self, context, model_input):
        if isinstance(model_input, dict):
            return self.model.predict_proba(pd.DataFrame(model_input))
        return self.model.predict_proba(model_input)

def parse_args():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str, help="Path to registered features data")
    parser.add_argument("--model_output", type=str, help="Path to model output")
    parser.add_argument("--model_name", type=str, default="eeg_classifier", help="Name under which model will be registered")
    args = parser.parse_args()
    return args

def load_and_validate_data(features_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """Load and validate feature data."""
    logger.info(f"Loading features from: {features_path}")
    
    features_file = Path(features_path) / "features.parquet"
    logger.info(f"Reading parquet file from: {features_file}")
    df = pd.read_parquet(features_file)
    
    # Log data distribution
    remission_counts = df.groupby('Participant')['Remission'].first().value_counts()
    logger.info("\nParticipant Remission Distribution:")
    logger.info(f"Number of Remission participants: {remission_counts.get(1, 0)}")
    logger.info(f"Number of Non-remission participants: {remission_counts.get(0, 0)}")
    
    groups = df['Participant'].values
    unique_participants = np.unique(groups)
    
    logger.info("\nParticipant-wise summary:")
    for participant in unique_participants:
        participant_data = df[df['Participant'] == participant]
        logger.info(f"\nParticipant {participant}:")
        logger.info(f"Samples: {len(participant_data)}")
        logger.info(f"Remission status: {'Yes' if participant_data['Remission'].iloc[0] == 1 else 'No'}")
    
    return df, unique_participants, groups

def train_and_evaluate(df: pd.DataFrame, groups: np.ndarray) -> Tuple[DecisionTreeClassifier, Dict, list]:
    """Train model with LOGO CV and return final model, metrics, and fold results."""
    X = df.drop(['Participant', 'Remission'], axis=1)
    y = df['Remission']
    
    # Calculate class weights based on class distribution
    class_counts = df.groupby('Participant')['Remission'].first().value_counts()
    total = class_counts.sum()
    class_weights = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    logger.info(f"\nClass weights: {class_weights}")
    
    # Initialize LOGO CV
    logo = LeaveOneGroupOut()
    fold_results = []
    
    logger.info("\nStarting Leave-One-Participant-Out Cross Validation:")
    
    # Perform CV
    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
        participant_id = groups[val_idx[0]]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model with class weights
        model = DecisionTreeClassifier(
            class_weight=class_weights,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Store results
        fold_metrics = {
            'participant': int(participant_id),
            'actual_class': int(y_val.iloc[0]),
            'predicted_class': int(val_pred[0]),
            'prediction_probability': float(val_pred_proba[0]),
            'samples': int(len(y_val)),
            'accuracy': float(accuracy_score(y_val, val_pred)),
            'precision': float(precision_score(y_val, val_pred, zero_division=0)),
            'recall': float(recall_score(y_val, val_pred, zero_division=0)),
            'f1': float(f1_score(y_val, val_pred, zero_division=0))
        }
        fold_results.append(fold_metrics)
        
        logger.info(f"\nParticipant {participant_id} (Fold {fold + 1}/{len(np.unique(groups))}):")
        logger.info(f"True class: {fold_metrics['actual_class']}")
        logger.info(f"Predicted class: {fold_metrics['predicted_class']}")
        logger.info(f"Prediction probability: {fold_metrics['prediction_probability']:.3f}")
        logger.info(f"Accuracy: {fold_metrics['accuracy']:.3f}")
    
    # Train final model on all data
    final_model = DecisionTreeClassifier(
        class_weight=class_weights,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    final_model.fit(X, y)
    
    # Calculate overall metrics
    results_df = pd.DataFrame(fold_results)
    metrics = {
        'accuracy_mean': float(results_df['accuracy'].mean()),
        'accuracy_std': float(results_df['accuracy'].std()),
        'precision_mean': float(results_df['precision'].mean()),
        'precision_std': float(results_df['precision'].std()),
        'recall_mean': float(results_df['recall'].mean()),
        'recall_std': float(results_df['recall'].std()),
        'f1_mean': float(results_df['f1'].mean()),
        'f1_std': float(results_df['f1'].std())
    }
    
    logger.info("\nOverall Cross-Validation Results:")
    logger.info(f"Accuracy: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
    logger.info(f"Precision: {metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f}")
    logger.info(f"Recall: {metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}")
    logger.info(f"F1: {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
    
    return final_model, metrics, fold_results

def save_training_results(model: DecisionTreeClassifier, df: pd.DataFrame, metrics: Dict, 
                         fold_results: list, output_path: Path):
    """Save training results and artifacts."""
    # Save fold-wise results
    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv(output_path / 'fold_results.csv', index=False)
    mlflow.log_artifact(str(output_path / 'fold_results.csv'))
    
    # Save feature importance
    feature_cols = df.drop(['Participant', 'Remission'], axis=1).columns
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': [float(imp) for imp in model.feature_importances_]
    }).sort_values('Importance', ascending=False)
    
    feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
    mlflow.log_artifact(str(output_path / 'feature_importance.csv'))
    
    # Log metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"cv_{metric_name}", value)
    
    # Log top feature importances
    for idx, row in feature_importance.head(10).iterrows():
        mlflow.log_metric(f"feature_importance_{row['Feature']}", row['Importance'])

def save_and_register_model(model, df, model_name, output_path):
    """Save and register model using MLflow."""
    # Create model signature
    input_example = df.drop(['Participant', 'Remission'], axis=1).iloc[:5]
    signature = infer_signature(
        df.drop(['Participant', 'Remission'], axis=1),
        model.predict(df.drop(['Participant', 'Remission'], axis=1))
    )
    
    # Save model in MLflow format directly to the output path
    # Note: Not using a 'model' subdirectory anymore
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(output_path),  # Save directly to output_path
        signature=signature,
        input_example=input_example
    )
    logger.info(f"Model saved locally to: {output_path}")
    
    # Log model to MLflow tracking
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
    
    # Register the model in Azure ML model registry
    registered_model = mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        model_name
    )
    
    return registered_model

def main():
    global start_time
    start_time = time.time()
    mlflow.start_run()
    args = parse_args()
    
    try:
        # Load data
        df, unique_participants, groups = load_and_validate_data(args.registered_features)
        
        # Train and evaluate model
        logger.info("\nStarting training...")
        logger.info(f"Number of participants: {len(unique_participants)}")
        
        training_start = time.time()
        model, metrics, fold_results = train_and_evaluate(df, groups)
        training_time = time.time() - training_start
        
        mlflow.log_metric("total_training_time", training_time)
        mlflow.log_metric("training_samples_per_second", len(df) / training_time)
        
        # Save results
        output_path = Path(args.model_output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_training_results(model, df, metrics, fold_results, output_path)
        
        # Save and register model
        registered_model = save_and_register_model(
            model=model,
            df=df,
            model_name=args.model_name,
            output_path=output_path
        )
        
        # Log final success metric
        mlflow.log_metric("training_success", 1)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        mlflow.log_metric("training_success", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()