import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import logging
from azureml.core import Run
from azureml.train.automl import AutoMLConfig
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_windows_to_patient(df):
    """
    Aggregate window-level features to patient-level features.
    """
    feature_cols = df.columns.difference(['Participant', 'Remission'])
    
    # Define aggregation functions
    agg_funcs = {col: ['mean', 'std', 'min', 'max', 'median'] 
                 for col in feature_cols}
    agg_funcs['Remission'] = 'first'
    
    # Aggregate
    patient_df = df.groupby('Participant').agg(agg_funcs)
    
    # Add percentiles
    percentiles = [25, 75]
    for col in feature_cols:
        for p in percentiles:
            patient_df[(col, f'percentile_{p}')] = df.groupby('Participant')[col].quantile(p/100)
    
    # Flatten column names
    patient_df.columns = [f"{col}_{agg}" if agg != 'first' else col 
                         for col, agg in patient_df.columns]
    
    # Add number of windows as a feature
    patient_df['n_windows'] = df.groupby('Participant').size()
    
    # Ensure all numeric columns are float64
    numeric_cols = patient_df.select_dtypes(include=['number']).columns
    patient_df[numeric_cols] = patient_df[numeric_cols].astype('float64')
    
    return patient_df.reset_index()

def get_automl_config(train_data, label_col='Remission', time_limit_minutes=15):
    """Create AutoML configuration"""
    return AutoMLConfig(
        task='classification',
        primary_metric='accuracy',
        training_data=train_data,
        label_column_name=label_col,
        n_cross_validations=None,
        validation_size=0,
        enable_early_stopping=True,
        experiment_timeout_minutes=time_limit_minutes,
        max_concurrent_iterations=4,
        max_cores_per_iteration=-1,
        enable_onnx_compatible_models=False,
        model_explainability=True,
        enable_ml_stats_collection=True,
        blocked_models=['TensorFlowDNN', 'TensorFlowLinearRegressor',
                       'LightGBMRegressor', 'TabularDeepLearning'],
        allowed_models=[
            'LogisticRegression', 'RandomForest', 'ExtremeRandomTrees',
            'GradientBoosting', 'XGBoostClassifier', 'SGDClassifier'
        ],
        enable_voting_ensemble=False,
        enable_stack_ensemble=False
    )

def parse_args():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str, help="Path to features dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--time_limit_minutes", type=int, default=15,
                       help="Time limit in minutes for AutoML optimization per fold")
    return parser.parse_args()

def main(args):
    # Get the experiment run context
    run = Run.get_context()
    
    logger.info(f"Loading training data from: {args.registered_features}")
    df = pd.read_parquet(Path(args.registered_features) / "features.parquet")
    
    # Aggregate to patient level
    logger.info("Aggregating window-level features to patient-level...")
    patient_df = aggregate_windows_to_patient(df)
    logger.info(f"Created {patient_df.shape[1]} patient-level features")
    
    # Prepare for training
    X = patient_df.drop(['Participant', 'Remission'], axis=1)
    y = patient_df['Remission']
    groups = patient_df['Participant']
    
    # Initialize LOGO cross-validation
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
    
    patient_results = []
    
    # Perform LOGO cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = groups.iloc[test_idx].iloc[0]
        true_label = y.iloc[test_idx].iloc[0]
        
        logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
        logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Apply SMOTE to training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Prepare training data for AutoML
        train_data = pd.concat([
            pd.DataFrame(X_train_resampled, columns=X.columns),
            pd.Series(y_train_resampled, name='Remission')
        ], axis=1)
        
        # Get AutoML config and train
        automl_config = get_automl_config(
            train_data=train_data,
            time_limit_minutes=args.time_limit_minutes
        )
        
        # Submit AutoML run as child run
        automl_run = run.submit_child(automl_config, show_output=True)
        automl_run.wait_for_completion(show_output=True)
        
        # Get the best model
        best_run, fitted_model = automl_run.get_output()
        
        # Make predictions
        pred_prob = fitted_model.predict_proba(X_test)[:, 1][0]
        pred_label = 1 if pred_prob >= 0.5 else 0
        
        # Store results
        patient_results.append({
            'participant': test_participant,
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': pred_prob,
            'correct_prediction': true_label == pred_label,
            'best_model': best_run.properties['algorithm_name']
        })
        
        # Log fold metrics
        mlflow.log_metric(f"fold_{fold_idx}_accuracy", int(patient_results[-1]['correct_prediction']))
        mlflow.log_metric(f"fold_{fold_idx}_confidence", patient_results[-1]['confidence'])
    
    # Calculate overall metrics
    patient_results_df = pd.DataFrame(patient_results)
    metrics = {
        'accuracy': accuracy_score(
            patient_results_df['true_label'],
            patient_results_df['predicted_label']
        ),
        'precision': precision_score(
            patient_results_df['true_label'],
            patient_results_df['predicted_label'],
            zero_division=0
        ),
        'recall': recall_score(
            patient_results_df['true_label'],
            patient_results_df['predicted_label'],
            zero_division=0
        ),
        'f1': f1_score(
            patient_results_df['true_label'],
            patient_results_df['predicted_label'],
            zero_division=0
        )
    }
    
    # Log metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"patient_level_{metric_name}", value)
    
    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    X_final_resampled, y_final_resampled = smote.fit_resample(X, y)
    
    # Prepare final training data
    final_train_data = pd.concat([
        pd.DataFrame(X_final_resampled, columns=X.columns),
        pd.Series(y_final_resampled, name='Remission')
    ], axis=1)
    
    final_automl_config = get_automl_config(
        train_data=final_train_data,
        time_limit_minutes=args.time_limit_minutes * 2
    )
    
    final_automl_run = run.submit_child(final_automl_config, show_output=True)
    final_automl_run.wait_for_completion(show_output=True)
    
    best_run, final_model = final_automl_run.get_output()
    
    # Save results
    patient_results_df.to_csv(Path(args.model_output) / 'patient_level_predictions.csv', index=False)
    
    # Save model using sklearn format
    signature = infer_signature(X, final_model.predict_proba(X)[:, 1])
    mlflow.sklearn.save_model(
        sk_model=final_model,
        path=args.model_output,
        signature=signature
    )
    
    # Log final model metrics
    final_metrics = final_automl_run.get_metrics()
    for metric_name, metric_value in final_metrics.items():
        mlflow.log_metric(f"final_model_{metric_name}", metric_value)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()