import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
import json
from azureml.train.automl import AutoMLConfig
from azureml.core import Run
from sklearn.model_selection import LeaveOneGroupOut
from mlflow.models.signature import infer_signature

class WrappedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
    
def parse_args():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--features_input", type=str, help="Path to features data")
    parser.add_argument("--model_output", type=str, help="Path to model output")
    args = parser.parse_args()
    return args

def main():
    mlflow.start_run()
    args = parse_args()
    run = Run.get_context()
    
    features_path = Path(args.features_input) / "features.parquet"
    df = pd.read_parquet(features_path)
    
    logo = LeaveOneGroupOut()
    groups = df['Participant'].values
    
    automl_config = AutoMLConfig(
        task='classification',
        primary_metric='AUC_weighted',
        training_data=df,
        label_column_name='Remission',
        compute_target=run.get_environment(),
        enable_early_stopping=True,
        experiment_timeout_minutes=15,
        max_concurrent_iterations=4,
        max_cores_per_iteration=-1,
        verbosity=logging.INFO,
        cv=logo,
        cv_groups=groups,
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
        featurization={
            'feature_normalization': 'MinMax',
            'drop_columns': ['Participant']
        }
    )

    print("Starting AutoML training...")
    print(f"Number of CV folds (participants): {len(np.unique(groups))}")
    automl_run = run.submit_child(automl_config, show_output=True)
    automl_run.wait_for_completion(show_output=True)

    best_run, fitted_model = automl_run.get_output()

    metrics = automl_run.get_metrics()
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    if hasattr(fitted_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': df.drop(['Participant', 'Remission'], axis=1).columns,
            'Importance': fitted_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance.to_csv(Path(args.model_output) / 'feature_importance.csv', index=False)
        mlflow.log_artifact(Path(args.model_output) / 'feature_importance.csv')

    cv_results = pd.DataFrame(automl_run.get_cv_results())
    cv_results.to_csv(Path(args.model_output) / 'cv_results.csv', index=False)
    mlflow.log_artifact(Path(args.model_output) / 'cv_results.csv')

    model_details = {
        'best_model_algorithm': type(fitted_model).__name__,
        'cv_folds': len(np.unique(groups)),
        'features_used': list(df.drop(['Participant', 'Remission'], axis=1).columns),
        'model_parameters': fitted_model.get_params()
    }
    
    with open(Path(args.model_output) / 'model_details.json', 'w') as f:
        json.dump(model_details, f, indent=2)
    mlflow.log_artifact(Path(args.model_output) / 'model_details.json')

    wrapped_model = WrappedModel(fitted_model)

    # Infer the model signature
    signature = infer_signature(df.drop(['Participant', 'Remission'], axis=1), fitted_model.predict(df.drop(['Participant', 'Remission'], axis=1)))

    # Save the model
    mlflow.pyfunc.save_model(
        path=args.model_output,
        python_model=wrapped_model,
        signature=signature
    )
    
    print(f"Model and results saved to: {args.model_output}")
    print("\nCross-validation metrics summary:")
    print(f"Mean AUC: {cv_results['AUC_weighted'].mean():.3f} ± {cv_results['AUC_weighted'].std():.3f}")
    print(f"Mean Accuracy: {cv_results['accuracy'].mean():.3f} ± {cv_results['accuracy'].std():.3f}")
    
    mlflow.end_run()

if __name__ == "__main__":
    main()