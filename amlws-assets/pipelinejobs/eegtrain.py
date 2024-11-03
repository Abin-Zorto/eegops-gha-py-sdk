import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command, dsl
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Training Pipeline")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument("--jobtype", type=str, help="Job Type")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--version", type=str, help="Version of registered features")
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args

def create_train_component(parent_dir, jobtype, environment_name):
    """Create the training component"""
    logger.info(f"Creating training component with environment: {environment_name}")
    return command(
        name="train_model_from_features",
        display_name="train-model-from-features",
        code=os.path.join(parent_dir, jobtype),
        command="python train_from_features.py \
                --registered_features ${{inputs.registered_features}} \
                --model_output ${{outputs.model_output}} \
                --model_name ${{inputs.model_name}}",
        environment=environment_name+"@latest",
        inputs={
            "registered_features": Input(type="mltable"),
            "model_name": Input(type="string")
        },
        outputs={
            "model_output": Output(type="uri_folder")
        }
    )

@dsl.pipeline(
    description="EEG Model Training Pipeline",
    display_name="EEG-Train-Pipeline"
)
def eeg_train_pipeline(registered_features, model_name):
    """Pipeline to train model"""
    logger.info("Initializing EEG training pipeline")
    
    # Training step
    logger.info("Setting up training job")
    train_job = train_model_from_features(
        registered_features=registered_features,
        model_name=model_name
    )
    
    return {
        "trained_model": train_job.outputs.model_output
    }

def main():
    logger.info("Starting training pipeline deployment")
    args = parse_args()
    
    # Set up Azure ML client
    logger.info("Setting up Azure ML client")
    credential = ClientSecretCredential(
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
        tenant_id=os.environ["AZURE_TENANT_ID"]
    )
    ml_client = MLClient.from_config(credential=credential)
    
    # Verify compute cluster
    try:
        compute_target = ml_client.compute.get(args.compute_name)
        logger.info(f"Found compute target: {compute_target.name}")
    except Exception as e:
        logger.error(f"Error accessing compute cluster: {str(e)}")
        raise
    
    parent_dir = "amlws-assets/src"
    logger.info(f"Using parent directory: {parent_dir}")
    
    # Create training component
    global train_model_from_features
    train_model_from_features = create_train_component(
        parent_dir, 
        args.jobtype, 
        args.environment_name
    )
    
    # Get the registered MLTable
    logger.info(f"Getting registered features version: {args.version}")
    registered_features = Input(type="mltable", path=f"azureml:eeg_features:{args.version}")
    
    # Create pipeline
    logger.info("Creating pipeline job")
    pipeline_job = eeg_train_pipeline(
        registered_features=registered_features,
        model_name=args.model_name
    )
    
    # Configure pipeline settings
    logger.info("Configuring pipeline settings")
    pipeline_job.settings.default_compute = args.compute_name
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.continue_on_step_failure = False
    pipeline_job.settings.force_rerun = True
    pipeline_job.settings.default_timeout = 3600
    
    # Submit and monitor pipeline job
    logger.info(f"Submitting pipeline job to experiment: {args.experiment_name}")
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )
    
    logger.info(f"Pipeline job submitted. Job name: {pipeline_job.name}")
    logger.info("Starting job monitoring...")
    ml_client.jobs.stream(pipeline_job.name)

if __name__ == "__main__":
    main()