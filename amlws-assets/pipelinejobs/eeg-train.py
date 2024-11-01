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
    return args

def setup_rai_components(ml_client_registry):
    """Set up RAI components from registry"""
    label = "latest"
    
    rai_constructor = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_insight_constructor", 
        label=label
    )
    version = rai_constructor.version
    
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
    return components

def create_train_component(parent_dir, jobtype, environment_name):
    return command(
        name="train_model_from_features",
        display_name="train-model-from-features",
        code=os.path.join(parent_dir, jobtype),
        command="python train_from_features.py \
                --registered_features ${{inputs.registered_features}} \
                --model_output ${{outputs.model_output}}",
        environment=environment_name+"@latest",
        inputs={
            "registered_features": Input(type="mltable")
        },
        outputs={
            "model_output": Output(type="uri_folder")
        }
    )

@dsl.pipeline(
    description="EEG Train Pipeline with RAI Dashboard",
    display_name="EEG-Train-Pipeline-RAI"
)
def eeg_train_pipeline(registered_features, rai_constructor, rai_error_analysis, rai_explanation, rai_gather, target_column_name="Remission"):
    # Training step
    train_job = train_model_from_features(
        registered_features=registered_features
    )
    
    # RAI dashboard construction
    create_rai_job = rai_constructor(
        title="EEG Analysis RAI Dashboard",
        task_type="classification",
        model_info="mlflow_model",
        model_input=train_job.outputs.model_output,
        train_dataset=registered_features,
        test_dataset=registered_features,  # Using same data since we're doing LOGO CV
        target_column_name=target_column_name,
    )
    create_rai_job.set_limits(timeout=300)
    
    # Add error analysis
    error_job = rai_error_analysis(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    error_job.set_limits(timeout=300)
    
    # Add explanations
    explanation_job = rai_explanation(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    explanation_job.set_limits(timeout=300)
    
    # Gather insights
    rai_gather_job = rai_gather(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_3=error_job.outputs.error_analysis,
        insight_4=explanation_job.outputs.explanation,
    )
    rai_gather_job.set_limits(timeout=300)
    # Set upload mode for dashboard
    rand_path = str(uuid.uuid4())
    rai_gather_job.outputs.dashboard = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
        mode="upload",
        type="uri_folder",
    )
    
    return {
        "trained_model": train_job.outputs.model_output,
        "rai_dashboard": rai_gather_job.outputs.dashboard
    }

def main():
    args = parse_args()
    print(args)
    
    credential = ClientSecretCredential(
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
        tenant_id=os.environ["AZURE_TENANT_ID"]
    )
    ml_client = MLClient.from_config(credential=credential)
    try:
        print(ml_client.compute.get(args.compute_name))
    except:
        print("No compute found")
    parent_dir = "amlws-assets/src"
    # Get RAI components
    registry_name = "azureml"
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        resource_group_name=ml_client.resource_group_name,
        registry_name=registry_name,
    )
    rai_components = setup_rai_components(ml_client_registry)
    # Create training component
    global train_model_from_features
    train_model_from_features = create_train_component(
        parent_dir, 
        args.jobtype, 
        args.environment_name
    )
    # Get the registered MLTable and create pipeline
    registered_features = Input(type="mltable", path=f"azureml:automl_features:{args.version}")
    
    pipeline_job = eeg_train_pipeline(
        registered_features=registered_features,
        rai_constructor=rai_components['constructor'],
        rai_error_analysis=rai_components['error_analysis'],
        rai_explanation=rai_components['explanation'],
        rai_gather=rai_components['gather']
    )
    # Set pipeline level compute
    pipeline_job.settings.default_compute = args.compute_name
    # Set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    # Add pipeline settings
    pipeline_job.settings.continue_on_step_failure = False
    pipeline_job.settings.force_rerun = True
    pipeline_job.settings.default_timeout = 3600
    # Submit and monitor pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )
    ml_client.jobs.stream(pipeline_job.name)

if __name__ == "__main__":
    main()