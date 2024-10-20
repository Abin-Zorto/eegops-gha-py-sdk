# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers trained ML model if deploy flag is True.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json
from ..train.train2 import WrappedModel  # Import WrappedModel from train.py

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    '''Loads model, registers it if deploy flag is True'''

    # Check deploy flag
    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    mlflow.log_metric("deploy flag", int(deploy_flag))

    # Proceed only if deploy flag is set
    if deploy_flag == 1:
        print(f"Registering {args.model_name}")

        # Load the model from the specified path
        model = mlflow.pyfunc.load_model(args.model_path)

        # Wrap the loaded model
        wrapped_model = WrappedModel(model)

        # Register the wrapped model using mlflow
        registered_model = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,  # Pass the wrapped model here
            registered_model_name=args.model_name
        )

        # Write model info as JSON
        print("Writing JSON")

        # Update this part to retrieve the version from registered_model instead of model
        model_info = {"id": f"{args.model_name}:{registered_model.version if hasattr(registered_model, 'version') else 'unknown'}"} 

        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(model_info, of)
    else:
        print("Model will not be registered!")

if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    # Print arguments for debugging
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Evaluation output path: {args.evaluation_output}")

    main(args)

    mlflow.end_run()
