import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import ResponsibleAIDashboard
import os
import json

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline with Responsible AI Dashboard")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument("--jobtype", type=str, help="Job Type")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--version", type=str, help="Version of registered features")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    credential = ClientSecretCredential(
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
        tenant_id=os.environ["AZURE_TENANT_ID"]
    )
    ml_client = MLClient.from_config(credential=credential)

    parent_dir = "amlws-assets/src"

    train_model_from_features = command(
        name="train_model_from_features",
        display_name="train-model-from-features",
        code=os.path.join(parent_dir, args.jobtype),
        command="python train_from_features.py \
                --registered_features ${{inputs.registered_features}} \
                --model_output ${{outputs.model_output}}",
        environment=args.environment_name + "@latest",
        inputs={
            "registered_features": Input(type="mltable", path=f"azureml:{args.data_name}:{args.version}")
        },
        outputs={
            "model_output": Output(type="uri_folder")
        }
    )

    responsible_ai_dashboard = ResponsibleAIDashboard(
        name="responsible_ai_dashboard",
        model=train_model_from_features.outputs.model_output,
        dataset=train_model_from_features.inputs.registered_features,
        compute_target=args.compute_name
    )

    @pipeline(
        description="EEG Train Pipeline with Responsible AI Dashboard",
        display_name="EEG-Train-Pipeline-with-RAI"
    )
    def eeg_train_pipeline():
        pipeline_job = train_model_from_features()
        responsible_ai_dashboard_job = responsible_ai_dashboard(
            model_output=pipeline_job.outputs.model_output,
            data=train_model_from_features.inputs.registered_features
        )
        return {"trained_model": pipeline_job.outputs.model_output}

    pipeline_job = eeg_train_pipeline()
    pipeline_job.settings.default_compute = args.compute_name
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.continue_on_step_failure = False
    pipeline_job.settings.force_rerun = True
    pipeline_job.settings.default_timeout = 3600

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )
    ml_client.jobs.stream(pipeline_job.name)

if __name__ == "__main__":
    main()
