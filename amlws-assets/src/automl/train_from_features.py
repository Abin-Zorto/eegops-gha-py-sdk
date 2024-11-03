import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
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
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading training data from: {features_path}")
    df = pd.read_parquet(features_path / "features.parquet")
    X = df.drop(["Participant", "Remission"], axis=1)
    y = df["Remission"]
    groups = df["Participant"]
    
    # Initialize LOGO cross-validation
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    logger.info(f"Performing Leave-One-Group-Out cross-validation with {n_splits} splits")
    
    # Initialize containers for results
    all_predictions = []
    fold_metrics = []
    
    # Log the unique groups
    unique_groups = np.unique(groups)
    logger.info(f"Total number of unique participants: {len(unique_groups)}")
    
    # Perform LOGO cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        # Log the split details
        test_group = groups.iloc[test_idx].unique()[0]
        logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
        logger.info(f"Test participant: {test_group}")
        logger.info(f"Train participants: {groups.iloc[train_idx].unique()}")
        logger.info(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        clf = RandomForestClassifier(random_state=42, min_samples_leaf=20)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        fold_metric = calculate_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(fold_metric)
        
        # Store predictions
        fold_results = pd.DataFrame({
            'participant': groups.iloc[test_idx],
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': y_prob,
            'fold': fold_idx
        })
        all_predictions.append(fold_results)
        
        # Log fold metrics
        logger.info("Fold metrics:")
        for metric_name, value in fold_metric.items():
            logger.info(f"    {metric_name}: {value:.3f}")
            mlflow.log_metric(f"fold_{fold_idx}_{metric_name}", value)
    
    # Combine all predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calculate and log overall metrics
    logger.info("\nOverall cross-validation metrics:")
    for metric in fold_metrics[0].keys():
        values = [m[metric] for m in fold_metrics]
        mean_value = np.mean(values)
        std_value = np.std(values)
        logger.info(f"{metric}:")
        logger.info(f"    Mean: {mean_value:.3f} (+/- {std_value:.3f})")
        mlflow.log_metric(f"cv_mean_{metric}", mean_value)
        mlflow.log_metric(f"cv_std_{metric}", std_value)
    
    # Per-participant analysis
    participant_metrics = all_predictions_df.groupby('participant').agg({
        'true_label': 'first',
        'predicted_label': 'first',
        'prediction_probability': 'first'
    })
    
    logger.info("\nPer-participant predictions:")
    for participant in participant_metrics.index:
        row = participant_metrics.loc[participant]
        logger.info(f"Participant {participant}:")
        logger.info(f"    True label: {row['true_label']}")
        logger.info(f"    Predicted: {row['predicted_label']} (probability: {row['prediction_probability']:.3f})")
    
    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    final_clf = RandomForestClassifier(random_state=42, min_samples_leaf=20)
    final_clf.fit(X, y)
    
    # Save model and predictions
    logger.info(f"Saving model and results to: {model_output_path}")
    
    # Save predictions
    all_predictions_df.to_csv(model_output_path / 'cross_validation_predictions.csv', index=False)
    
    # Save final model
    signature = mlflow.models.infer_signature(X, final_clf.predict_proba(X)[:, 1])
    mlflow.sklearn.save_model(
        sk_model=final_clf,
        path=model_output_path,
        signature=signature
    )
    
    mlflow.sklearn.log_model(
        sk_model=final_clf,
        artifact_path="model",
        registered_model_name=args.model_name
    )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()