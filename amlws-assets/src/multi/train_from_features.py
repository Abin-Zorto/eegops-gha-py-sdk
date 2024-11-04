import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import logging
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_patient_prediction(window_predictions, threshold=0.5):
    """
    Convert window-level predictions to a patient-level prediction.
    Returns both the majority vote prediction and the confidence (proportion of windows predicted as positive)
    """
    proportion_positive = np.mean(window_predictions)
    prediction = 1 if proportion_positive >= threshold else 0
    return prediction, proportion_positive

def get_classifiers():
    """Return dictionary of sklearn classifiers to try"""
    return {
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=42
            ))
        ]),
        'gradient_boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                random_state=42
            ))
        ]),
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])
    }

def parse_args():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str, help="Path to features dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_name", type=str, help="Model name")
    return parser.parse_args()

def main(args):
    logger.info(f"Loading training data from: {args.registered_features}")
    df = pd.read_parquet(Path(args.registered_features) / "features.parquet")
    
    # Prepare for training
    X = df.drop(['Participant', 'Remission'], axis=1)
    y = df['Remission']
    groups = df['Participant']
    
    # Calculate and log class distribution
    unique_patients = groups.unique()
    patient_labels = df.groupby('Participant')['Remission'].first()
    n_remission_patients = sum(patient_labels == 1)
    n_non_remission_patients = sum(patient_labels == 0)
    
    # Log detailed dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total number of patients: {len(unique_patients)}")
    logger.info(f"- Remission patients: {n_remission_patients}")
    logger.info(f"- Non-remission patients: {n_non_remission_patients}")
    logger.info(f"Total number of windows: {len(df)}")
    
    # Log windows per patient statistics
    windows_per_patient = df.groupby('Participant').size()
    logger.info("\nWindows per patient:")
    logger.info(f"- Mean: {windows_per_patient.mean():.1f}")
    logger.info(f"- Min: {windows_per_patient.min()}")
    logger.info(f"- Max: {windows_per_patient.max()}")
    logger.info(f"- Median: {windows_per_patient.median()}")
    
    # Get classifiers to try
    classifiers = get_classifiers()
    
    # Initialize LOGO cross-validation
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
    
    # Dictionary to store results for each classifier
    all_patient_results = {name: [] for name in classifiers.keys()}
    all_window_results = {name: [] for name in classifiers.keys()}
    
    # Perform LOGO cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = groups.iloc[test_idx].unique()[0]
        true_label = y.iloc[test_idx].iloc[0]  # All windows for a patient have same label
        
        logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
        logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        logger.info(f"Training windows: {len(X_train)}")
        logger.info(f"Test windows: {len(X_test)}")
        
        # Try each classifier
        for name, clf in classifiers.items():
            logger.info(f"\nTraining {name}")
            
            # Train the model
            clf.fit(X_train, y_train)
            
            # Make predictions on all windows
            window_probs = clf.predict_proba(X_test)[:, 1]
            window_preds = clf.predict(X_test)
            
            # Calculate patient-level prediction
            patient_pred, confidence = calculate_patient_prediction(window_preds)
            
            # Store window-level results
            all_window_results[name].append(pd.DataFrame({
                'participant': test_participant,
                'true_label': true_label,
                'window_prediction': window_preds,
                'window_probability': window_probs,
                'fold': fold_idx
            }))
            
            # Store patient-level results
            all_patient_results[name].append({
                'participant': test_participant,
                'true_label': true_label,
                'predicted_label': patient_pred,
                'confidence': confidence,
                'n_windows': len(test_idx),
                'n_windows_positive': sum(window_preds == 1),
                'n_windows_negative': sum(window_preds == 0),
                'proportion_positive': confidence,
                'correct_prediction': true_label == patient_pred
            })
            
            # Log fold metrics
            mlflow.log_metric(f"{name}_fold_{fold_idx}_accuracy", 
                            int(all_patient_results[name][-1]['correct_prediction']))
            mlflow.log_metric(f"{name}_fold_{fold_idx}_confidence", 
                            all_patient_results[name][-1]['confidence'])
    
    # Calculate and log overall metrics for each classifier
    best_classifier = None
    best_f1 = -1
    
    for name in classifiers.keys():
        # Combine results
        patient_results_df = pd.DataFrame(all_patient_results[name])
        window_results_df = pd.concat(all_window_results[name], ignore_index=True)
        
        # Calculate metrics
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
        logger.info(f"\nMetrics for {name}:")
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{name}_{metric_name}", value)
            logger.info(f"{metric_name}: {value:.3f}")
        
        # Save results
        patient_results_df.to_csv(Path(args.model_output) / f'{name}_patient_level_predictions.csv', 
                                index=False)
        window_results_df.to_csv(Path(args.model_output) / f'{name}_window_level_predictions.csv', 
                               index=False)
        
        # Track best classifier based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_classifier = name
    
    logger.info(f"\nBest performing classifier: {best_classifier} (F1: {best_f1:.3f})")
    
    # Train final model on all data using best classifier
    logger.info("\nTraining final model on all data...")
    final_model = classifiers[best_classifier]
    final_model.fit(X, y)
    
    # Save best model using sklearn format
    signature = infer_signature(X, final_model.predict_proba(X)[:, 1])
    mlflow.sklearn.save_model(
        sk_model=final_model,
        path=args.model_output,
        signature=signature
    )
    
    # Log model parameters
    for param_name, param_value in final_model.named_steps['clf'].get_params().items():
        mlflow.log_param(f"best_model_{param_name}", param_value)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()