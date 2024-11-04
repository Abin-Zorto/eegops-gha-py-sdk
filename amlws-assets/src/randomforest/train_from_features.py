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

def aggregate_windows_to_patient(df):
    """
    Aggregate window-level features to patient-level features.
    """
    feature_cols = df.columns.difference(['Participant', 'Remission'])
    
    # Define aggregation functions
    # Ensure zero_crossings features are treated as numeric
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
        
        # Aggregate to patient level first
        logger.info("Aggregating window-level features to patient-level...")
        patient_df = aggregate_windows_to_patient(df)
        logger.info(f"Created {patient_df.shape[1]} patient-level features")
        
        # After aggregating to patient level
        logger.info("Checking for missing values...")
        
        # Count missing values per column
        missing_counts = patient_df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if len(columns_with_missing) > 0:
            logger.warning(f"\nFound {len(columns_with_missing)} columns with missing values:")
            for col, count in columns_with_missing.items():
                logger.warning(f"{col}: {count} missing values")
                
            # Check if missing values are from specific participants
            rows_with_missing = patient_df[patient_df.isnull().any(axis=1)]
            if not rows_with_missing.empty:
                logger.warning(f"\nParticipants with missing values:")
                for participant in rows_with_missing['Participant'].unique():
                    missing_cols = patient_df[patient_df['Participant'] == participant].isnull().sum()
                    missing_cols = missing_cols[missing_cols > 0]
                    logger.warning(f"Participant {participant}: {len(missing_cols)} columns with missing values")
        
        # Prepare for training
        X = patient_df.drop(['Participant', 'Remission'], axis=1)
        y = patient_df['Remission']
        groups = patient_df['Participant']  # for LOGO CV
        
        # Calculate class weights
        n_samples = len(y)
        n_remission = sum(y == 1)
        n_non_remission = n_samples - n_remission
        
        class_weights = {
            0: 1,
            1: n_non_remission / n_remission
        }
        
        logger.info(f"Class distribution - Remission: {n_remission}, Non-remission: {n_non_remission}")
        logger.info(f"Using class weights: {class_weights}")
        
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
            
            # Train model
            clf = RandomForestClassifier(
                random_state=42,
                min_samples_leaf=2,  # Reduced since we have fewer samples now
                n_estimators=200,
                max_depth=10,
                class_weight=class_weights,
                bootstrap=True,
                max_features='sqrt'
            )
            clf.fit(X_train_resampled, y_train_resampled)
            
            # Make predictions
            pred_prob = clf.predict_proba(X_test)[0, 1]  # Single patient prediction
            pred_label = 1 if pred_prob >= 0.5 else 0  # Using same threshold
            
            # Store results
            patient_results.append({
                'participant': test_participant,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': pred_prob,
                'correct_prediction': true_label == pred_label
            })
            
            logger.info(f"Patient prediction: {pred_label} (confidence: {pred_prob:.3f}, true: {true_label})")
        
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
            logger.info(f"{metric_name.capitalize()}: {value:.3f}")
        
        # Train final model on all data
        logger.info("\nTraining final model on all data...")
        X_final_resampled, y_final_resampled = smote.fit_resample(X, y)
        
        final_clf = RandomForestClassifier(
            random_state=42,
            min_samples_leaf=2,
            n_estimators=200,
            max_depth=10,
            class_weight=class_weights,
            bootstrap=True,
            max_features='sqrt'
        )
        final_clf.fit(X_final_resampled, y_final_resampled)
        
        # Save results and model
        patient_results_df.to_csv(model_output_path / 'patient_level_predictions.csv', index=False)
        
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': final_clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv(model_output_path / 'feature_importance.csv', index=False)
        
        logger.info("\nTop 10 most important features:")
        for _, row in feature_importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        # Save final model
        signature = mlflow.models.infer_signature(X, final_clf.predict_proba(X)[:, 1])
        mlflow.sklearn.log_model(
            sk_model=final_clf,
            artifact_path="model",
            registered_model_name=args.model_name,
            signature=signature
        )
        
        logger.info("Training completed successfully")

if __name__ == "__main__":
    main()