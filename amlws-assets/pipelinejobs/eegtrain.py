import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command, dsl
import os
import json
import uuid
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline")
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

def setup_rai_components(ml_client_registry):
    """Set up RAI components from registry"""
    logger.info("Setting up RAI components...")
    label = "latest"
    
    rai_constructor = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_insight_constructor", 
        label=label
    )
    version = rai_constructor.version
    logger.info(f"Using RAI components version: {version}")
    
    components = {
        'constructor': rai_constructor,
        'error_analysis': ml_client_registry.components.get(
            name="microsoft_azureml_rai_tabular_erroranalysis", 
            version=version
        ),
        'explanation': ml_client_registry.components.get(
            name="microsoft_azureml_rai_tabular_explanation", 
            version=version
        ),
        'gather': ml_client_registry.components.get(
            name="microsoft_azureml_rai_tabular_insight_gather", 
            version=version
        )
    }
    logger.info("RAI components setup complete")
    return components

def create_train_component(parent_dir, jobtype, environment_name):
    logger.info(f"Creating training component with environment: {environment_name}")
    return command(
        name="train_model_from_features",
        display_name="train-model-from-features",
        code=os.path.join(parent_dir, jobtype),
        command="python train_from_features.py \
                --registered_features ${{inputs.registered_features}} \
                --model_output ${{outputs.model_output}} \
                --model_name ${{inputs.model_name}}",  # Added model_name as input
        environment=environment_name+"@latest",
        inputs={
            "registered_features": Input(type="mltable"),
            "model_name": Input(type="string")  # Added model_name input
        },
        outputs={
            "model_output": Output(type="uri_folder")
        }
    )

@dsl.pipeline(
    description="EEG Train Pipeline with RAI Dashboard",
    display_name="EEG-Train-Pipeline-RAI"
)
def eeg_train_pipeline(registered_features, model_name, target_column_name="Remission"):
    """Pipeline to train model and generate RAI dashboard"""
    logger.info("Initializing EEG training pipeline")
    
    # Training step
    logger.info("Setting up training job")
    train_job = train_model_from_features(
        registered_features=registered_features,
        model_name=model_name
    )
    
    # Log paths for debugging
    logger.info(f"Train job output path: {train_job.outputs.model_output}")
    
    # Add detailed logging for model paths
    logger.info(f"Train job output structure:")
    logger.info(f"- Full path: {train_job.outputs.model_output}")
    logger.info(f"- Output name: {train_job.outputs.model_output.name}")
    logger.info(f"- Output path: {train_job.outputs.model_output.path}")
    
    # RAI dashboard construction
    logger.info("Setting up RAI constructor job")
    logger.info(f"RAI constructor model input path (before): {train_job.outputs.model_output}")
    
    create_rai_job = rai_constructor(
        title="EEG Analysis RAI Dashboard",
        task_type="classification",
        model_info="mlflow_model",
        model_input=train_job.outputs.model_output,
        train_dataset=registered_features,
        test_dataset=registered_features,
        target_column_name=target_column_name,
    )
    
    logger.info(f"RAI constructor job created with:")
    logger.info(f"- Model input: {create_rai_job.inputs.model_input}")
    
    # Changed: Log the RAI job itself instead of trying to access its inputs
    logger.info(f"RAI constructor job created: {create_rai_job.name}")
    
    create_rai_job.set_limits(timeout=300)
    
    # Rest of the pipeline remains the same...
    error_job = rai_error_analysis(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    error_job.set_limits(timeout=300)
    
    explanation_job = rai_explanation(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        comment="Feature importance and SHAP values for EEG classification model"
    )
    explanation_job.set_limits(timeout=300)
    
    rai_gather_job = rai_gather(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_3=error_job.outputs.error_analysis,
        insight_4=explanation_job.outputs.explanation,
    )
    rai_gather_job.set_limits(timeout=300)
    
    rand_path = str(uuid.uuid4())
    dashboard_path = f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/"
    logger.info(f"Dashboard will be saved to: {dashboard_path}")
    
    rai_gather_job.outputs.dashboard = Output(
        path=dashboard_path,
        mode="upload",
        type="uri_folder",
    )
    
    return {
        "trained_model": train_job.outputs.model_output,
        "rai_dashboard": rai_gather_job.outputs.dashboard
    }

def main():
    logger.info("Starting pipeline deployment")
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
        logger.error("No compute found")
        raise
    
    parent_dir = "amlws-assets/src"
    logger.info(f"Using parent directory: {parent_dir}")
    
    # Get RAI components from registry
    registry_name = "azureml"
    logger.info(f"Accessing registry: {registry_name}")
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        resource_group_name=ml_client.resource_group_name,
        registry_name=registry_name,
    )
    
    # Setup components
    rai_components = setup_rai_components(ml_client_registry)
    
    # Create training component and assign RAI components
    logger.info("Creating pipeline components")
    global train_model_from_features
    global rai_constructor, rai_error_analysis, rai_explanation, rai_gather
    
    train_model_from_features = create_train_component(
        parent_dir, 
        args.jobtype, 
        args.environment_name
    )
    rai_constructor = rai_components['constructor']
    rai_error_analysis = rai_components['error_analysis']
    rai_explanation = rai_components['explanation']
    rai_gather = rai_components['gather']
    
    # Get the registered MLTable
    logger.info(f"Getting registered features version: {args.version}")
    registered_features = Input(type="mltable", path=f"azureml:automl_features:{args.version}")
    
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