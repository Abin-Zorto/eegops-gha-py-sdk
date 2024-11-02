import argparse
import os
import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Register Trained Model")
    parser.add_argument("--model_path", type=str, help="Path to the trained model directory")
    parser.add_argument("--model_name", type=str, help="Name to register the model as")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, help="Azure resource group")
    parser.add_argument("--workspace_name", type=str, help="Azure ML workspace name")
    parser.add_argument("--client_id", type=str, help="Azure client ID")
    parser.add_argument("--client_secret", type=str, help="Azure client secret")
    parser.add_argument("--tenant_id", type=str, help="Azure tenant ID")
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args

def main():
    args = parse_args()
    
    # Initialize MLClient
    logger.info("Initializing MLClient...")
    credential = ClientSecretCredential(
        client_id=args.client_id,
        client_secret=args.client_secret,
        tenant_id=args.tenant_id
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    # Register the model
    logger.info(f"Registering model '{args.model_name}' from path '{args.model_path}'...")
    registered_model = ml_client.models.create_or_update(
        Model(
            path=args.model_path,
            name=args.model_name,
            type="mlflow_pyfunc",
            description="Registered model for EEG classification",
            tags={"framework": "scikit-learn", "task": "classification"}
        )
    )
    
    logger.info(f"Model registered successfully with name: {registered_model.name} and version: {registered_model.version}")

if __name__ == "__main__":
    main() 