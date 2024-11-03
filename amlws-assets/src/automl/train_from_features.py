import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime  # Import datetime for timestamp

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

def calculate_patient_metrics(y_true, y_pred):
    """Calculate classification metrics at the patient level"""
    return {
        'accuracy': accuracy_score([y_true], [y_pred]),
        'correct_prediction': y_true == y_pred
    }

def main():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str)
    parser.add_argument("--model_output", type=str)
    parser.add_argument("--model_name", type=str, default="automl")
    args = parser.parse_args()
    
    logger.info(f"Received arguments: {args}")
    
    features_path = Path(args.registered_features)
    model_output_path = Path(args.model_output)
    
    # **Append timestamp to model_output to ensure uniqueness**
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_model_output = model_output_path / f"model_{timestamp}"
    
    logger.info(f"Using unique model output path: {unique_model_output}")
    
    unique_model_output.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading training data from: {features_path}")
    df = pd.read_parquet(features_path / "features.parquet")
    X = df.drop(["Participant", "Remission"], axis=1)
    y = df["Remission"]
    groups = df["Participant"]
    
    # Analyze patient distribution
    patient_stats = []
    for participant in groups.unique():
        mask = groups == participant
        windows = mask.sum()
        label = y[mask].iloc[0]  # All windows for a patient have same label
        patient_stats.append({
            'participant': participant,
            'n_windows': windows,
            'label': label
        })
    
    patient_df = pd.DataFrame(patient_stats)
    logger.info("\nPatient Statistics:")
    logger.info(f"Total patients: {len(patient_df)}")
    logger.info(f"Remission patients: {(patient_df['label'] == 1).sum()}")
    logger.info(f"Non-remission patients: {(patient_df['label'] == 0).sum()}")
    logger.info("\nWindows per patient:")
    logger.info(f"Mean: {patient_df['n_windows'].mean():.1f}")
    logger.info(f"Min: {patient_df['n_windows'].min()}")
    logger.info(f"Max: {patient_df['n_windows'].max()}")
    
    # Initialize LOGO cross-validation
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
    
    # Initialize containers for results
    patient_results = []
    window_results = []
    
    # Perform LOGO cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_participant = groups.iloc[test_idx].unique()[0]
        true_label = y.iloc[test_idx].iloc[0]  # All windows have same label
        
        logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
        logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        clf = RandomForestClassifier(random_state=42, min_samples_leaf=20)
        clf.fit(X_train, y_train)
        
        # Make predictions on all windows
        window_probs = clf.predict_proba(X_test)[:, 1]
        window_preds = clf.predict(X_test)
        
        # Calculate patient-level prediction
        patient_pred, confidence = calculate_patient_prediction(window_preds)
        
        # Store window-level results
        window_results.append(pd.DataFrame({
            'participant': test_participant,
            'true_label': true_label,
            'window_prediction': window_preds,
            'window_probability': window_probs,
            'fold': fold_idx
        }))
        
        # Store patient-level results
        patient_result = {
            'participant': test_participant,
            'true_label': true_label,
            'predicted_label': patient_pred,
            'confidence': confidence,
            'n_windows': len(test_idx),
            'n_windows_positive': sum(window_preds == 1),
            'n_windows_negative': sum(window_preds == 0),
            'proportion_positive': confidence
        }
        patient_results.append(patient_result)
        
        # Log fold results
        logger.info(f"Windows predicted positive: {patient_result['n_windows_positive']}/{patient_result['n_windows']}" 
                   f" ({patient_result['proportion_positive']:.1%})")
        logger.info(f"Patient-level prediction: {patient_pred} (true: {true_label})")
    
    # Combine all results
    patient_results_df = pd.DataFrame(patient_results)
    window_results_df = pd.concat(window_results, ignore_index=True)
    
    # Calculate overall metrics
    patient_accuracy = accuracy_score(
        patient_results_df['true_label'], 
        patient_results_df['predicted_label']
    )
    patient_precision = precision_score(
        patient_results_df['true_label'], 
        patient_results_df['predicted_label'],
        zero_division=0
    )
    patient_recall = recall_score(
        patient_results_df['true_label'], 
        patient_results_df['predicted_label'],
        zero_division=0
    )
    patient_f1 = f1_score(
        patient_results_df['true_label'], 
        patient_results_df['predicted_label'],
        zero_division=0
    )
    
    logger.info("\nOverall Patient-Level Metrics:")
    logger.info(f"Accuracy: {patient_accuracy:.3f}")
    logger.info(f"Precision: {patient_precision:.3f}")
    logger.info(f"Recall: {patient_recall:.3f}")
    logger.info(f"F1 Score: {patient_f1:.3f}")
    
    # Log misclassified patients
    misclassified = patient_results_df[
        patient_results_df['true_label'] != patient_results_df['predicted_label']
    ]
    logger.info(f"\nMisclassified Patients ({len(misclassified)}/{len(patient_results_df)}):")
    for _, row in misclassified.iterrows():
        logger.info(
            f"Participant {row['participant']}: "
            f"True={row['true_label']}, "
            f"Pred={row['predicted_label']}, "
            f"Confidence={row['confidence']:.1%}, "
            f"Windows={row['n_windows']}"
        )
    
    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    final_clf = RandomForestClassifier(random_state=42, min_samples_leaf=20)
    final_clf.fit(X, y)
    
    # Save results
    logger.info(f"Saving model and results to: {unique_model_output}")
    
    # Save predictions
    patient_results_df.to_csv(unique_model_output / 'patient_level_predictions.csv', index=False)
    window_results_df.to_csv(unique_model_output / 'window_level_predictions.csv', index=False)
    
    # Save feature importances
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance_df.to_csv(unique_model_output / 'feature_importance.csv', index=False)
    
    # Log top features
    logger.info("\nTop 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save final model
    signature = mlflow.models.infer_signature(X, final_clf.predict_proba(X)[:, 1])
    mlflow.sklearn.save_model(
        sk_model=final_clf,
        path=unique_model_output,
        signature=signature
    )
    
    # **Log and Register the Model with MLflow**
    mlflow.sklearn.log_model(
        sk_model=final_clf,
        artifact_path="model",
        registered_model_name=args.model_name  # This will handle versioning automatically
    )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()