# register_features.py
import argparse
from pathlib import Path
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
import os
import mlflow
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("register_features")
    parser.add_argument("--features_input", type=str, help="Path to features parquet file")
    parser.add_argument("--data_name", type=str, help="Name for registered data asset")
    parser.add_argument("--description", type=str, default="EEG features for depression classification")
    parser.add_argument("--registered_features_output", type=str, help="Path to save registered features info")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, help="Azure resource group name")
    parser.add_argument("--workspace_name", type=str, help="Azure ML workspace name")
    args = parser.parse_args()
    return args

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        
        # Initialize MLClient
        logger.info("Initializing MLClient...")
        credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"]
        )
        ml_client = MLClient(
            credential=credential,
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name
        )
        
        logger.info(f"Registering features from: {args.features_input}")
        
        # Create MLTable definition
        mltable_data = Data(
            path=args.features_input,
            type=AssetTypes.MLTABLE,
            description=args.description,
            name=args.data_name,
            version='1.0.0'
        )
        
        # Register the data
        registered_data = ml_client.data.create_or_update(mltable_data)
        
        # Save registration information
        registration_info = {
            "name": registered_data.name,
            "version": registered_data.version,
            "id": registered_data.id,
            "path": str(Path(args.features_input))
        }
        
        output_path = Path(args.registered_features_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(registration_info, f)
        
        # Log registration metrics
        mlflow.log_metric("registration_time", time.time() - start_time)
        mlflow.log_metric("registration_status", 1)
        
        logger.info(f"Features registered as MLTable: {registered_data.name}, version: {registered_data.version}")
        
    except Exception as e:
        logger.error(f"Error registering features: {str(e)}")
        mlflow.log_metric("registration_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()