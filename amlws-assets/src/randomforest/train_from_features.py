import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_patient_prediction(window_predictions, threshold=0.3):  # Lowered threshold
    """
    Convert window-level predictions to a patient-level prediction.
    Returns both the majority vote prediction and the confidence (proportion of windows predicted as positive)
    """
    proportion_positive = np.mean(window_predictions)
    prediction = 1 if proportion_positive >= threshold else 0
    return prediction, proportion_positive

def apply_smote_to_windows(X_train, y_train):
    """Apply SMOTE to window-level data"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Original class distribution: {Counter(y_train)}")
    logger.info(f"Resampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def main():
    parser = argparse.ArgumentParser("train_from_features")
    parser.add_argument("--registered_features", type=str)
    parser.add_argument("--model_output", type=str)
    parser.add_argument("--model_name", type=str, default="automl")
    args = parser.parse_args()
    
    with mlflow.start_run():
        logger.info(f"Received arguments: {args}")
        
        features_path = Path(args.registered_features)
        model_output_path = Path(args.model_output)
        model_output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        logger.info(f"Loading training data from: {features_path}")
        df = pd.read_parquet(features_path / "features.parquet")
        X = df.drop(["Participant", "Remission"], axis=1)
        y = df["Remission"]
        groups = df["Participant"]
        
        # Calculate class weights
        n_samples = len(groups.unique())
        n_remission = sum(y.iloc[groups.drop_duplicates().index] == 1)
        n_non_remission = n_samples - n_remission
        
        class_weights = {
            0: 1,
            1: n_non_remission / n_remission  # Give higher weight to minority class
        }
        
        logger.info(f"Using class weights: {class_weights}")
        
        # Initialize LOGO cross-validation
        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(X, y, groups)
        logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
        
        patient_results = []
        window_results = []
        
        # Perform LOGO cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_participant = groups.iloc[test_idx].unique()[0]
            true_label = y.iloc[test_idx].iloc[0]
            
            logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
            logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Apply SMOTE to training data
            X_train_resampled, y_train_resampled = apply_smote_to_windows(X_train, y_train)
            
            # Train model with balanced class weights and adjusted parameters
            clf = RandomForestClassifier(
                random_state=42,
                min_samples_leaf=5,  # Reduced from 20
                n_estimators=200,    # Increased from default
                max_depth=10,        # Prevent overfitting
                class_weight=class_weights,
                bootstrap=True,
                max_features='sqrt'  # Helps prevent overfitting
            )
            clf.fit(X_train_resampled, y_train_resampled)
            
            # Make predictions
            window_probs = clf.predict_proba(X_test)[:, 1]
            # Use probability threshold of 0.3 for window-level predictions
            window_preds = (window_probs >= 0.3).astype(int)
            
            # Calculate patient-level prediction
            patient_pred, confidence = calculate_patient_prediction(window_preds)
            
            # Store results
            window_results.append(pd.DataFrame({
                'participant': test_participant,
                'true_label': true_label,
                'window_prediction': window_preds,
                'window_probability': window_probs,
                'fold': fold_idx
            }))
            
            patient_results.append({
                'participant': test_participant,
                'true_label': true_label,
                'predicted_label': patient_pred,
                'confidence': confidence,
                'n_windows': len(test_idx),
                'n_windows_positive': sum(window_preds == 1),
                'proportion_positive': confidence,
                'correct_prediction': true_label == patient_pred
            })
            
            logger.info(f"Windows predicted positive: {sum(window_preds == 1)}/{len(window_preds)}"
                       f" ({confidence:.1%})")
            logger.info(f"Patient-level prediction: {patient_pred} (true: {true_label})")
        
        # Calculate overall metrics
        patient_results_df = pd.DataFrame(patient_results)
        window_results_df = pd.concat(window_results, ignore_index=True)
        
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
            logger.info(f"{metric_name.capitalize()}: {value:.3f}")
        
        # Train final model on all data
        logger.info("\nTraining final model on all data...")
        X_final_resampled, y_final_resampled = apply_smote_to_windows(X, y)
        
        final_clf = RandomForestClassifier(
            random_state=42,
            min_samples_leaf=5,
            n_estimators=200,
            max_depth=10,
            class_weight=class_weights,
            bootstrap=True,
            max_features='sqrt'
        )
        final_clf.fit(X_final_resampled, y_final_resampled)
        
        # Save results and model
        patient_results_df.to_csv(model_output_path / 'patient_level_predictions.csv', index=False)
        window_results_df.to_csv(model_output_path / 'window_level_predictions.csv', index=False)
        
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': final_clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv(model_output_path / 'feature_importance.csv', index=False)
        
        logger.info("\nTop 10 most important features:")
        for _, row in feature_importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        mlflow.sklearn.log_model(
            sk_model=final_clf,
            artifact_path="model",
            registered_model_name=args.model_name
        )
        
        logger.info("Training completed successfully")

if __name__ == "__main__":
    main()