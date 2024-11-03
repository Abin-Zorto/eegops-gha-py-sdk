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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, required=True, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, required=True, help="Data Asset Name")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name")
    parser.add_argument("--version", type=str, required=True, help="Version of registered features")
    parser.add_argument("--model_version", type=str, required=False, default="latest", help="Version of the registered model")
    parser.add_argument("--environment_name", type=str, required=True, help="Environment name")
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

def create_split_component(environment_name):
    """Create the data splitting component"""
    return command(
        name="split_data",
        display_name="split-data",
        code="amlws-assets/src",
        command="python split_data.py \
                --input_mltable ${{inputs.input_mltable}} \
                --train_data ${{outputs.train_data}} \
                --test_data ${{outputs.test_data}} \
                --test_size 0.2",
        environment=environment_name+"@latest",
        inputs={
            "input_mltable": Input(type="mltable")
        },
        outputs={
            "train_data": Output(type="mltable"),
            "test_data": Output(type="mltable")
        }
    )

def create_rai_pipeline(
    compute_name: str,
    model_name: str,
    model_version: str,
    target_column_name: str,
    input_data: Input,
    environment_name: str,
    rai_components: Dict
):
    """Create the RAI pipeline with all components"""
    # Create the split component
    split_data = create_split_component(environment_name)
    
    @dsl.pipeline(
        compute=compute_name,
        description="RAI insights on EEG data",
        experiment_name=f"RAI_insights_{model_name}",
    )
    def rai_decision_pipeline(
        target_column_name, input_data
    ):
        # Split the data
        split_job = split_data(
            input_mltable=input_data
        )
        
        expected_model_id = f"{model_name}:{model_version}"
        azureml_model_id = f"azureml:{expected_model_id}"
        
        create_rai_job = rai_components['constructor'](
            title="RAI dashboard EEG",
            task_type="classification",
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=split_job.outputs.train_data,
            test_dataset=split_job.outputs.test_data,
            target_column_name=target_column_name,
            classes='[false, true]',
            categorical_column_names=['Remission']
        )
        create_rai_job.set_limits(timeout=300)

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
        input_data=input_data
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
        registered_features = Input(
            type="mltable",
            path=f"azureml:{args.data_name}:{args.version}",
            mode="ro_mount"
        )
        
        # Create pipeline
        logger.info("Creating RAI pipeline job")
        pipeline_job = create_rai_pipeline(
            compute_name=args.compute_name,
            model_name=args.model_name,
            model_version=args.model_version,
            target_column_name="Remission",
            input_data=registered_features,
            environment_name=args.environment_name,
            rai_components=rai_components
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