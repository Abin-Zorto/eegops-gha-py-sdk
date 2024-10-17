# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import Environment, BuildContext

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Register environment")
    parser.add_argument("--environment_name", type=str, required=True, help="Name of the environment")
    parser.add_argument("--description", type=str, required=True, help="Description of the environment")
    parser.add_argument("--env_path", type=str, required=True, help="Path to the environment file")
    parser.add_argument("--build_type", type=str, required=True, choices=['docker', 'conda'], help="Build type for the environment")
    parser.add_argument("--base_image", type=str, default="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04", help="Base image for the environment")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Arguments: {args}")
    
    try:
        # Initialize MLClient
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential)

        # Create the environment
        env = Environment(
            name=args.environment_name,
            description=args.description,
            image=args.base_image,
            conda_file=args.env_path if args.build_type == 'conda' else None,
            dockerfile_path=args.env_path if args.build_type == 'docker' else None
        )

        # Register the environment
        registered_env = ml_client.environments.create_or_update(env)
        print(f"Environment registered: {registered_env.name}, version: {registered_env.version}")

    except Exception as ex:
        print(f"An error occurred: {str(ex)}")
        raise

if __name__ == "__main__":
    main()
