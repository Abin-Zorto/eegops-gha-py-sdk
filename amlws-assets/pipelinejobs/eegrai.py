import argparse
from azure.ai.ml.entities import Data, Model
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

@dsl.pipeline(
    description="EEG RAI Dashboard Generation Pipeline",
    display_name="EEG-RAI-Dashboard"
)
def eeg_rai_pipeline(
    registered_features, 
    model_name
):
    """Pipeline to generate RAI dashboard using registered model"""
    logger.info("Initializing EEG RAI pipeline")
    
    # RAI dashboard construction using registered model
    logger.info("Setting up RAI constructor job")
    create_rai_job = rai_constructor(
        title="EEG RAI Analysis",
        task_type="classification",
        model_info=Input(
            type="mlflow_model",
            path=f"azureml:{model_name}:2"
        ),
        train_dataset=registered_features,
        test_dataset=registered_features,
        target_column_name="Remission",
        categorical_column_names="[]",
        classes='["Non-remission", "Remission"]'
    )
    
    create_rai_job.set_limits(timeout=300)
    
    # Error analysis
    error_job = rai_error_analysis(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        max_depth=3,  # Optional: depth of the error analysis tree
        num_leaves=31,  # Optional: number of leaves in the error analysis tree
        min_child_samples=20  # Optional: minimum samples required at a leaf node
    )
    error_job.set_limits(timeout=300)
    
    # Model explanation
    explanation_job = rai_explanation(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        comment="Model explanations for EEG classification"
    )
    explanation_job.set_limits(timeout=300)
    
    # Gather insights
    rai_gather_job = rai_gather(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_1=error_job.outputs.error_analysis,
        insight_2=explanation_job.outputs.explanation
    )
    rai_gather_job.set_limits(timeout=300)
    
    # Set up dashboard output path
    rand_path = str(uuid.uuid4())
    dashboard_path = f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/"
    logger.info(f"Dashboard will be saved to: {dashboard_path}")
    
    rai_gather_job.outputs.dashboard = Output(
        path=dashboard_path,
        mode="upload",
        type="uri_folder",
    )
    
    return {
        "rai_dashboard": rai_gather_job.outputs.dashboard,
        "rai_insights": create_rai_job.outputs.rai_insights_dashboard
    }

def main():
    logger.info("Starting RAI pipeline deployment")
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
    
    # Verify model exists
    try:
        model = ml_client.models.get(args.model_name, label="latest")
        logger.info(f"Found model {args.model_name} (version {model.version})")
    except Exception as e:
        logger.error(f"Error accessing model: {str(e)}")
        raise
    
    # Get RAI components from registry
    registry_name = "azureml"
    logger.info(f"Accessing registry: {registry_name}")
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        resource_group_name=ml_client.resource_group_name,
        registry_name=registry_name,
    )
    
    # Setup RAI components
    rai_components = setup_rai_components(ml_client_registry)
    
    # Assign RAI components
    global rai_constructor, rai_error_analysis, rai_explanation, rai_gather
    rai_constructor = rai_components['constructor']
    rai_error_analysis = rai_components['error_analysis']
    rai_explanation = rai_components['explanation']
    rai_gather = rai_components['gather']
    
    # Get the registered MLTable
    logger.info(f"Getting registered features version: {args.version}")
    registered_features = Input(
        type="mltable",
        path=f"azureml:{args.data_name}:{args.version}",
        mode="ro_mount"
    )
    
    # Create pipeline
    logger.info("Creating RAI pipeline job")
    pipeline_job = eeg_rai_pipeline(
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