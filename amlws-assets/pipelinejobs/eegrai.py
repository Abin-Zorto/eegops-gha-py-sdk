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
from typing import Dict
from azure.ai.ml import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_modify_mltable(ml_client, data_name, version, drop_columns):
    """Load MLTable, drop specified columns, and save as new MLTable."""
    
    # Load the dataset as a DataFrame
    dataset = Dataset.get_by_name(ml_client.workspace, name=data_name, version=version)
    df = dataset.to_pandas_dataframe()
    
    # Drop specified columns
    df.drop(columns=drop_columns, inplace=True)
    logger.info(f"Dropped columns: {drop_columns}")
    
    # Register the modified DataFrame as a new MLTable
    modified_data_name = f"{data_name}_modified"
    modified_dataset = Dataset.Tabular.register_pandas_dataframe(df, ml_client.workspace, modified_data_name)
    logger.info(f"Registered modified dataset: {modified_data_name}")
    
    # Return new MLTable as Input type for pipeline
    return Input(type="mltable", path=f"azureml:{modified_data_name}:{version}", mode="ro_mount")


def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, required=True, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, required=True, help="Data Asset Name")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name")
    parser.add_argument("--version", type=str, required=True, help="Version of registered features")
    parser.add_argument("--model_version", type=str, required=False, default="latest", help="Version of the registered model")
    return parser.parse_args()

def setup_rai_components(ml_client_registry):
    """Set up RAI components from registry
    
    Args:
        ml_client_registry: Azure ML registry client
        
    Returns:
        dict: Dictionary containing RAI components
    """
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

def create_rai_pipeline(
    compute_name: str,
    model_name: str,
    model_version: str,
    target_column_name: str,
    train_data: Input,
    test_data: Input,
    rai_components: Dict
):
    """Create the RAI pipeline with all components
    
    Args:
        compute_name: Name of the compute cluster
        model_name: Name of the registered model
        target_column_name: Name of the target column
        train_data: Training data input
        test_data: Test data input
        rai_components: Dictionary of RAI components
    
    Returns:
        pipeline: Configured RAI pipeline
    """
    @dsl.pipeline(
        compute=compute_name,
        description="RAI insights on EEG data",
        experiment_name=f"RAI_insights_{model_name}",
    )
    def rai_decision_pipeline(
        target_column_name, train_data, test_data
    ):
        expected_model_id = f"{model_name}:{model_version}"
        azureml_model_id = f"azureml:{expected_model_id}"
        
        logger.info(f"Using model ID: {expected_model_id}")
        logger.info(f"Azure ML Model URI: {azureml_model_id}")
        
        create_rai_job = rai_components['constructor'](
            title="RAI dashboard EEG",
            task_type="classification",
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
            categorical_column_names='[]',  # Remove 'Participant' from categorical columns
            classes='["Non-remission", "Remission"]',
            feature_metadata='{"dropped_features": ["Participant"]}'  # Add feature metadata to drop Participant column
        )
        create_rai_job.set_limits(timeout=300)

        logger.info("RAI Insights job initialized successfully.")

        error_job = rai_components['error_analysis'](
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        error_job.set_limits(timeout=300)

        explanation_job = rai_components['explanation'](
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="add explanation", 
        )
        explanation_job.set_limits(timeout=300)

        gather_job = rai_components['gather'](
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_3=error_job.outputs.error_analysis,
            insight_4=explanation_job.outputs.explanation,
        )
        gather_job.set_limits(timeout=300)

        gather_job.outputs.dashboard.mode = "upload"

        return {
            "dashboard": gather_job.outputs.dashboard,
        }
    
    return rai_decision_pipeline(
        target_column_name=target_column_name,
        train_data=train_data,
        test_data=test_data
    )

def main():
    try:
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
        compute_target = ml_client.compute.get(args.compute_name)
        logger.info(f"Found compute target: {compute_target.name}")
        
        # Verify model exists
        model = ml_client.models.get(args.model_name, version=args.model_version)
        logger.info(f"Found model {args.model_name} (version {model.version}) with URI: {model.path}")
        
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
        
        # Get the registered MLTable
        logger.info(f"Getting registered features version: {args.version}")
        modified_features = load_and_modify_mltable(
            ml_client=ml_client,
            data_name=args.data_name,
            version=args.version,
            drop_columns=["Participant"]  # Specify columns to drop
        )
            
        # Create pipeline
        logger.info("Creating RAI pipeline job")
        pipeline_job = create_rai_pipeline(
            compute_name=args.compute_name,
            model_name=args.model_name,
            model_version=args.model_version,
            target_column_name="Remission",
            train_data=modified_features,
            test_data=modified_features,
            rai_components=setup_rai_components(ml_client_registry)
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
        
    except Exception as e:
        logger.error(f"Pipeline deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()