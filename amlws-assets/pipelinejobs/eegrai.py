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
    parser = argparse.ArgumentParser("Deploy RAI Insights Pipeline")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--model_name", type=str, help="Registered Model Name")
    parser.add_argument("--model_version", type=str, default="latest", help="Registered Model Version")
    parser.add_argument("--target_column_name", type=str, help="Name of the target column")
    parser.add_argument("--train_data", type=str, help="Path to training data MLTable")
    parser.add_argument("--test_data", type=str, help="Path to testing data MLTable")
    parser.add_argument("--experiment_name", type=str, help="Azure ML Experiment Name")
    return parser.parse_args()

def setup_rai_components(ml_client_registry):
    # Implementation to setup RAI components
    # This typically involves preparing the necessary components like constructor, error_analysis, etc.
    # Placeholder for actual implementation
    components = {
        'constructor': ml_client_registry.components.get(name="rai-dashboard-constructor"),
        'error_analysis': ml_client_registry.components.get(name="rai-error-analysis"),
        'explanation': ml_client_registry.components.get(name="rai-explanation"),
        'gather': ml_client_registry.components.get(name="rai-gather")
    }
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
    @dsl.pipeline(
        compute=compute_name,
        description="RAI Insights on EEG Data",
        experiment_name=f"RAI_insights_{model_name}",
    )
    def rai_decision_pipeline(
        target_column_name, train_data, test_data
    ):
        args = parse_args()
        expected_model_id = f"{model_name}:{model_version}"
        azureml_model_id = f"azureml:{expected_model_id}"
        
        logger.info(f"Using model ID: {expected_model_id}")
        logger.info(f"Azure ML Model URI: {azureml_model_id}")
        
        # Initiate the RAIInsights
        create_rai_job = rai_components['constructor'](
            title="RAI Dashboard EEG",
            task_type="classification",
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
            categorical_column_names='["Participant"]',
            classes='["Non-remission", "Remission"]'
        )
        create_rai_job.set_limits(timeout=300)

        # Error Analysis
        error_job = rai_components['error_analysis'](
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        error_job.set_limits(timeout=300)

        # Explanation
        explanation_job = rai_components['explanation'](
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="add explanation", 
        )
        explanation_job.set_limits(timeout=300)

        # Gather Insights
        gather_job = rai_components['gather'](
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_3=error_job.outputs.error_analysis,
            insight_4=explanation_job.outputs.explanation,
        )
        gather_job.set_limits(timeout=300)

        # Upload Dashboard
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
        logger.info("Starting RAI Insights Pipeline Deployment")
        args = parse_args()
        
        # Initialize MLClient
        logger.info("Setting up Azure ML client")
        credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"]
        )
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
        )
        
        # Verify compute cluster
        compute_target = ml_client.compute.get(args.compute_name)
        logger.info(f"Using compute target: {compute_target.name}")
        
        # Verify model exists
        if args.model_version.lower() == "latest":
            model = ml_client.models.get(name=args.model_name, latest=True)
        else:
            model = ml_client.models.get(name=args.model_name, version=args.model_version)
        logger.info(f"Using model: {model.name}, version: {model.version}, path: {model.path}")
        
        # Setup RAI Components
        registry = ml_client.registry
        rai_components = setup_rai_components(registry)
        
        # Define training and testing data inputs
        train_input = Input(
            type="mltable",
            path=args.train_data,
            mode="ro_mount"
        )
        test_input = Input(
            type="mltable",
            path=args.test_data,
            mode="ro_mount"
        )
        
        # Create RAI Pipeline
        logger.info("Creating RAI Insights pipeline")
        pipeline_job = create_rai_pipeline(
            compute_name=args.compute_name,
            model_name=args.model_name,
            model_version=model.version,
            target_column_name=args.target_column_name,
            train_data=train_input,
            test_data=test_input,
            rai_components=rai_components
        )
        
        # Submit the pipeline job
        logger.info(f"Submitting pipeline job to experiment: {args.experiment_name}")
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name=args.experiment_name
        )
        
        # Stream the job logs
        logger.info(f"Streaming job logs for pipeline: {pipeline_job.name}")
        ml_client.jobs.stream(pipeline_job.name)
        logger.info("RAI Insights pipeline execution completed successfully.")
    
    except Exception as e:
        logger.error(f"RAI Insights pipeline deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()