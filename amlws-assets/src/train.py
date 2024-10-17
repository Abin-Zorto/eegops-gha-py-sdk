# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using AutoML and training dataset. Saves trained model using MLflow pyfunc.
"""

import argparse
from pathlib import Path
import pandas as pd
from azureml.core import Run
from azureml.train.automl import AutoMLConfig
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

class WrappedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--modelname", type=str, help="model name")
    args = parser.parse_args()
    return args

def main(args):
    # Get the experiment run context
    run = Run.get_context()

    # Load the training data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into inputs and outputs
    y_train = train_data['cost']
    X_train = train_data.drop('cost', axis=1)

    # Configure AutoML
    automl_config = AutoMLConfig(
        task='regression',
        primary_metric='r2_score',
        training_data=train_data,
        label_column_name='cost',
        n_cross_validations=5,
        enable_early_stopping=True,
        experiment_timeout_minutes=15,
        max_concurrent_iterations=4,
        max_cores_per_iteration=-1,
        enable_onnx_compatible_models=False,
        blocked_models=['TensorFlowDNN', 'TensorFlowLinearRegressor'],
        allowed_models=[
            'RandomForest', 'LightGBM', 'DecisionTree',
            'ElasticNet', 'GradientBoosting', 'KNN',
            'FastLinearRegressor', 'LassoLars', 'SGDRegressor', 'ExtraTreesRegressor'
        ]
    )

    # Submit the AutoML experiment
    automl_run = run.submit_child(automl_config, show_output=True)

    # Wait for the run to complete
    automl_run.wait_for_completion(show_output=True)

    # Get the best model
    best_run, fitted_model = automl_run.get_output()

    # Wrap the model
    wrapped_model = WrappedModel(fitted_model)

    # Infer the model signature
    signature = infer_signature(X_train, fitted_model.predict(X_train))


    # Save the model to the specified output path
    mlflow.pyfunc.save_model(
        path=args.model_output,
        python_model=wrapped_model,
        signature=signature
    )

    # Log metrics
    metrics = automl_run.get_metrics()
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    print(f"Best model saved to {args.model_output}")

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()