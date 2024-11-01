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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WrappedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def parse_args():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str, help="Path to registered features data")
    parser.add_argument("--model_output", type=str, help="Path to model output")
    args = parser.parse_args()
    return args

def load_and_validate_data(features_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """Load and validate feature data."""
    logger.info(f"Loading features from: {features_path}")
    
    # Load parquet file directly
    features_file = Path(features_path) / "features.parquet"
    logger.info(f"Reading parquet file from: {features_file}")
    df = pd.read_parquet(features_file)
    
    # Compute data statistics
    stats = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 2,  # excluding Participant and Remission
        'unique_participants': len(df['Participant'].unique()),
        'class_balance': df['Remission'].value_counts(normalize=True).to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Log data statistics
    for name, value in stats.items():
        if isinstance(value, dict):
            for k, v in value.items():
                mlflow.log_metric(f"data_{name}_{k}", v)
        else:
            mlflow.log_metric(f"data_{name}", value)
    
    groups = df['Participant'].values
    
    return df, stats, groups

def train_and_evaluate(df: pd.DataFrame, groups: np.ndarray) -> Tuple[DecisionTreeClassifier, Dict[str, float]]:
    """Train and evaluate model using LOGO CV."""
    X = df.drop(['Participant', 'Remission'], axis=1)
    y = df['Remission']
    
    cv = LeaveOneGroupOut()
    cv_metrics = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Train final model on all data
    final_model = DecisionTreeClassifier(random_state=42)
    final_model.fit(X, y)
    
    # Perform CV for evaluation
    logger.info("Starting Leave-One-Group-Out Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        cv_metrics['auc'].append(roc_auc_score(y_val, y_pred_proba))
        cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_metrics['precision'].append(precision_score(y_val, y_pred))
        cv_metrics['recall'].append(recall_score(y_val, y_pred))
        cv_metrics['f1'].append(f1_score(y_val, y_pred))
        
        logger.info(f"Fold {fold + 1}/{len(np.unique(groups))}: AUC = {cv_metrics['auc'][-1]:.3f}")
    
    # Calculate mean and std of metrics
    metrics_summary = {}
    for metric in cv_metrics:
        values = cv_metrics[metric]
        metrics_summary[f'{metric}_mean'] = np.mean(values)
        metrics_summary[f'{metric}_std'] = np.std(values)
        
        # Log metrics to MLflow
        mlflow.log_metric(f'cv_{metric}_mean', metrics_summary[f'{metric}_mean'])
        mlflow.log_metric(f'cv_{metric}_std', metrics_summary[f'{metric}_std'])
    
    return final_model, metrics_summary

def save_training_results(model: DecisionTreeClassifier, df: pd.DataFrame, metrics: Dict[str, float], output_path: Path):
    """Save training results and artifacts."""
    feature_cols = df.drop(['Participant', 'Remission'], axis=1).columns
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
    mlflow.log_artifact(output_path / 'feature_importance.csv')
    
    for idx, row in feature_importance.head(10).iterrows():
        mlflow.log_metric(f"top_feature_{idx+1}_importance", row['Importance'])
    
    model_details = {
        'model_type': 'DecisionTreeClassifier',
        'cv_folds': len(np.unique(df['Participant'])),
        'features_used': list(df.drop(['Participant', 'Remission'], axis=1).columns),
        'model_parameters': model.get_params(),
        'cross_validation_metrics': metrics,
        'training_time': time.time() - start_time
    }
    
    with open(output_path / 'model_details.json', 'w') as f:
        json.dump(model_details, f, indent=2)
    mlflow.log_artifact(output_path / 'model_details.json')

def main():
    global start_time
    start_time = time.time()
    mlflow.start_run()
    args = parse_args()
    
    try:
        # Load data
        df, data_stats, groups = load_and_validate_data(args.registered_features)
        
        # Train and evaluate model
        logger.info("Starting training...")
        logger.info(f"Number of participants (CV folds): {len(np.unique(groups))}")
        
        training_start = time.time()
        model, metrics = train_and_evaluate(df, groups)
        training_time = time.time() - training_start
        
        mlflow.log_metric("total_training_time", training_time)
        mlflow.log_metric("training_samples_per_second", len(df) / training_time)
        
        # Save results
        output_path = Path(args.model_output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_training_results(model, df, metrics, output_path)
        
        # Save model
        wrapped_model = WrappedModel(model)
        signature = infer_signature(
            df.drop(['Participant', 'Remission'], axis=1),
            model.predict(df.drop(['Participant', 'Remission'], axis=1))
        )
        mlflow.pyfunc.save_model(
            path=args.model_output,
            python_model=wrapped_model,
            signature=signature
        )
        
        # Log summary metrics
        logger.info("\nCross-validation metrics summary:")
        logger.info(f"Mean AUC: {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
        logger.info(f"Mean Accuracy: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
        
        # Log final success metric
        mlflow.log_metric("training_success", 1)
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        mlflow.log_metric("training_success", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()