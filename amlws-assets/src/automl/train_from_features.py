import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    parser.add_argument("--time_limit", type=int, default=300, 
                       help="Time limit in seconds for AutoML optimization per fold")
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
        
        # Log patient distribution metrics
        mlflow.log_metric("total_patients", len(patient_df))
        mlflow.log_metric("remission_patients", (patient_df['label'] == 1).sum())
        mlflow.log_metric("non_remission_patients", (patient_df['label'] == 0).sum())
        mlflow.log_metric("mean_windows_per_patient", patient_df['n_windows'].mean())
        mlflow.log_metric("min_windows_per_patient", patient_df['n_windows'].min())
        mlflow.log_metric("max_windows_per_patient", patient_df['n_windows'].max())
        
        # Initialize LOGO cross-validation
        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(X, y, groups)
        logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
        
        # Initialize containers for results
        patient_results = []
        window_results = []
        fold_metrics = []
        
        # Perform LOGO cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_participant = groups.iloc[test_idx].unique()[0]
            true_label = y.iloc[test_idx].iloc[0]  # All windows have same label
            
            logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
            logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Initialize and train AutoML model
            automl = AutoSklearnClassifier(
                time_left_for_this_task=args.time_limit,
                per_run_time_limit=args.time_limit//10,
                ensemble_size=1,  # To ensure we get a single best model
                metric=accuracy_score,
                include_estimators=["random_forest", "extra_trees", "gradient_boosting", "sgd", 
                                  "adaboost", "xgradient_boosting"],  # sklearn-compatible models
                include_preprocessors=["no_preprocessing", "select_percentile", "pca"],
                resampling_strategy='holdout',
                resampling_strategy_arguments={'train_size': 1},
                random_state=42
            )
            
            automl.fit(X_train, y_train)
            
            # Make predictions on all windows
            window_probs = automl.predict_proba(X_test)[:, 1]
            window_preds = automl.predict(X_test)
            
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
                'proportion_positive': confidence,
                'correct_prediction': true_label == patient_pred,
                'best_model': automl.show_models().iloc[0]['type']  # Log best model type
            }
            patient_results.append(patient_result)
            
            # Log fold metrics
            mlflow.log_metric(f"fold_{fold_idx}_accuracy", int(patient_result['correct_prediction']))
            mlflow.log_metric(f"fold_{fold_idx}_confidence", patient_result['confidence'])
            mlflow.log_metric(f"fold_{fold_idx}_positive_window_ratio", 
                            patient_result['n_windows_positive'] / patient_result['n_windows'])
            
            # Log fold results
            logger.info(f"Best model for fold: {patient_result['best_model']}")
            logger.info(f"Windows predicted positive: {patient_result['n_windows_positive']}/{patient_result['n_windows']}" 
                       f" ({patient_result['proportion_positive']:.1%})")
            logger.info(f"Patient-level prediction: {patient_pred} (true: {true_label})")
        
        # Combine all results
        patient_results_df = pd.DataFrame(patient_results)
        window_results_df = pd.concat(window_results, ignore_index=True)
        
        # Calculate and log overall metrics
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
        
        # Log overall performance metrics
        mlflow.log_metric("patient_level_accuracy", patient_accuracy)
        mlflow.log_metric("patient_level_precision", patient_precision)
        mlflow.log_metric("patient_level_recall", patient_recall)
        mlflow.log_metric("patient_level_f1", patient_f1)
        
        logger.info("\nOverall Patient-Level Metrics:")
        logger.info(f"Accuracy: {patient_accuracy:.3f}")
        logger.info(f"Precision: {patient_precision:.3f}")
        logger.info(f"Recall: {patient_recall:.3f}")
        logger.info(f"F1 Score: {patient_f1:.3f}")
        
        # Log misclassified patients
        misclassified = patient_results_df[
            patient_results_df['true_label'] != patient_results_df['predicted_label']
        ]
        n_misclassified = len(misclassified)
        mlflow.log_metric("misclassified_patients", n_misclassified)
        mlflow.log_metric("misclassification_rate", n_misclassified / len(patient_results_df))
        
        logger.info(f"\nMisclassified Patients ({n_misclassified}/{len(patient_results_df)}):")
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
        final_automl = AutoSklearnClassifier(
            time_left_for_this_task=args.time_limit * 2,  # Give more time for final model
            per_run_time_limit=args.time_limit//5,
            ensemble_size=1,
            metric=accuracy_score,
            include_estimators=["random_forest", "extra_trees", "gradient_boosting", "sgd", 
                              "adaboost", "xgradient_boosting"],
            include_preprocessors=["no_preprocessing", "select_percentile", "pca"],
            resampling_strategy='holdout',
            resampling_strategy_arguments={'train_size': 1},
            random_state=42
        )
        final_automl.fit(X, y)
        
        # Save predictions
        logger.info(f"Saving results to: {model_output_path}")
        patient_results_df.to_csv(model_output_path / 'patient_level_predictions.csv', index=False)
        window_results_df.to_csv(model_output_path / 'window_level_predictions.csv', index=False)
        
        # Log best model details
        best_model_stats = final_automl.show_models().iloc[0]
        mlflow.log_params({
            "best_model_type": best_model_stats['type'],
            "best_model_accuracy": best_model_stats['cost'],
            "automl_time_limit": args.time_limit,
        })
        
        # Save leaderboard
        leaderboard = final_automl.leaderboard()
        leaderboard.to_csv(model_output_path / 'model_leaderboard.csv', index=False)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=final_automl,
            artifact_path="model",
            registered_model_name=args.model_name
        )
        
        # Log CSV files as artifacts
        mlflow.log_artifact(str(model_output_path / 'patient_level_predictions.csv'))
        mlflow.log_artifact(str(model_output_path / 'model_leaderboard.csv'))
        
        logger.info("Training completed successfully")

if __name__ == "__main__":
    main()