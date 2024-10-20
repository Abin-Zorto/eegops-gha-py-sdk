import argparse
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Register environment")
    parser.add_argument("--environment_name", type=str, required=True, help="Name of the environment")
    parser.add_argument("--description", type=str, required=True, help="Description of the environment")
    parser.add_argument("--env_path", type=str, required=True, help="Path to the environment file")
    parser.add_argument("--build_type", type=str, required=True, default="conda", help="Build type for the environment")
    parser.add_argument("--base_image", type=str, default="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04", help="Base image for the environment")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Arguments: {args}")
    
    try:
        # Initialize MLClient
        credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"]
        )
        ml_client = MLClient.from_config(credential=credential)

        # Create the environment
        env_docker_conda = Environment(
            image=args.base_image,
            conda_file=args.env_path,
            name=args.environment_name,
            description=args.description,
        )
        ml_client.environments.create_or_update(env_docker_conda)

    except Exception as ex:
        print(f"An error occurred: {str(ex)}")
        raise

if __name__ == "__main__":
    main()
