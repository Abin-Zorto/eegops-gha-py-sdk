import argparse
from pathlib import Path
import pandas as pd
import mlflow
import logging
import time
import tempfile
from azureml.train.automl import AutoMLConfig
from azureml.core import Run, Dataset
from sklearn.model_selection import LeaveOneGroupOut
from mlflow.models.signature import infer_signature
from typing import Dict, Any, Tuple
import json
import numpy as np
import os

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
    parser.add_argument("--experiment_timeout", type=int, default=15, help="Experiment timeout in minutes")
    args = parser.parse_args()
    return args

def load_and_validate_data(features_path: str, run: Run) -> Tuple[Dataset, pd.DataFrame, Dict[str, Any], np.ndarray]:
    """Load and validate feature data from mounted path."""
    logger.info(f"Loading features from: {features_path}")
    
    # Load parquet file directly
    features_file = Path(features_path) / "features.parquet"
    logger.info(f"Reading parquet file from: {features_file}")
    df = pd.read_parquet(features_file)
    
    # Convert DataFrame to Dataset
    workspace = run.experiment.workspace
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save DataFrame temporarily as CSV in the temp directory
        temp_csv = os.path.join(temp_dir, "temp_features.csv")
        logger.info(f"Creating temporary CSV file at: {temp_csv}")
        df.to_csv(temp_csv, index=False)
        
        # Create Dataset from the temp CSV file
        dataset = Dataset.Tabular.from_delimited_files(
            path=[(None, temp_csv)],
            validate=True
        )
        
        # File will be automatically cleaned up when the context manager exits
    
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
    
    return dataset, df, stats, groups

def create_automl_config(dataset: Dataset, groups: np.ndarray, run: Run) -> AutoMLConfig:
    """Create AutoML configuration with comprehensive settings."""
    samples_per_participant = len(dataset.to_pandas_dataframe()) / len(np.unique(groups))
    logger.info(f"Average samples per participant: {samples_per_participant:.2f}")
    
    return AutoMLConfig(
        task='classification',
        primary_metric='AUC_weighted',
        training_data=dataset,
        label_column_name='Remission',
        compute_target=run.get_environment(),
        enable_early_stopping=True,
        experiment_timeout_minutes=15,
        iteration_timeout_minutes=5,
        max_concurrent_iterations=4,
        max_cores_per_iteration=-1,
        verbosity=logging.INFO,
        cv=LeaveOneGroupOut(),
        cv_groups=groups,
        n_cross_validations=None,
        blocked_models=['TensorFlowDNN', 'TensorFlowLinearClassifier'],
        allowed_models=[
            'LogisticRegression',
            'RandomForest',
            'LightGBM',
            'XGBoostClassifier',
            'ExtremeRandomTrees',
            'GradientBoosting'
        ],
        model_explainability=True,
        enable_onnx_compatible_models=False,
        featurization='auto'
    )

def save_training_results(fitted_model: Any, df: pd.DataFrame, automl_run: Any, output_path: Path):
    """Save comprehensive training results and artifacts."""
    if hasattr(fitted_model, 'feature_importances_'):
        feature_cols = df.drop(['Participant', 'Remission'], axis=1).columns
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': fitted_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
        mlflow.log_artifact(output_path / 'feature_importance.csv')
        
        for idx, row in feature_importance.head(10).iterrows():
            mlflow.log_metric(f"top_feature_{idx+1}_importance", row['Importance'])

    cv_results = pd.DataFrame(automl_run.get_cv_results())
    cv_results.to_csv(output_path / 'cv_results.csv', index=False)
    mlflow.log_artifact(output_path / 'cv_results.csv')
    
    cv_metrics = {
        'auc_mean': cv_results['AUC_weighted'].mean(),
        'auc_std': cv_results['AUC_weighted'].std(),
        'accuracy_mean': cv_results['accuracy'].mean(),
        'accuracy_std': cv_results['accuracy'].std(),
        'precision_mean': cv_results['precision'].mean(),
        'precision_std': cv_results['precision'].std(),
        'recall_mean': cv_results['recall'].mean(),
        'recall_std': cv_results['recall'].std(),
        'f1_mean': cv_results['f1-score'].mean(),
        'f1_std': cv_results['f1-score'].std()
    }
    for metric_name, value in cv_metrics.items():
        mlflow.log_metric(f"cv_{metric_name}", value)

    model_details = {
        'best_model_algorithm': type(fitted_model).__name__,
        'cv_folds': len(np.unique(df['Participant'])),
        'features_used': list(df.drop(['Participant', 'Remission'], axis=1).columns),
        'model_parameters': fitted_model.get_params(),
        'cross_validation_metrics': cv_metrics,
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
    run = Run.get_context()
    
    try:
        # Load dataset and convert to DataFrame for additional processing
        dataset, df, data_stats, groups = load_and_validate_data(args.registered_features, run)
        
        automl_config = create_automl_config(dataset, groups, run)
        logger.info("Starting AutoML training...")
        logger.info(f"Number of CV folds (participants): {len(np.unique(groups))}")
        
        training_start = time.time()
        automl_run = run.submit_child(automl_config, show_output=True)
        automl_run.wait_for_completion(show_output=True)
        training_time = time.time() - training_start
        
        best_run, fitted_model = automl_run.get_output()
        
        mlflow.log_metric("total_training_time", training_time)
        mlflow.log_metric("training_samples_per_second", len(df) / training_time)
        
        output_path = Path(args.model_output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_training_results(fitted_model, df, automl_run, output_path)
        
        wrapped_model = WrappedModel(fitted_model)
        signature = infer_signature(
            df.drop(['Participant', 'Remission'], axis=1),
            fitted_model.predict(df.drop(['Participant', 'Remission'], axis=1))
        )
        mlflow.pyfunc.save_model(
            path=args.model_output,
            python_model=wrapped_model,
            signature=signature
        )
        
        logger.info(f"Model and results saved to: {args.model_output}")
        logger.info("\nCross-validation metrics summary:")
        cv_results = pd.DataFrame(automl_run.get_cv_results())
        logger.info(f"Mean AUC: {cv_results['AUC_weighted'].mean():.3f} ± {cv_results['AUC_weighted'].std():.3f}")
        logger.info(f"Mean Accuracy: {cv_results['accuracy'].mean():.3f} ± {cv_results['accuracy'].std():.3f}")
        
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