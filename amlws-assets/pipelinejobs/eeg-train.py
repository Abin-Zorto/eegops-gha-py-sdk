import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.dsl import pipeline
from azure.identity import ClientSecretCredential
from azure.ai.ml.constants import AssetTypes
import os

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline with Responsible AI Dashboard")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--version", type=str, help="Version of registered features")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    credential = ClientSecretCredential(
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
        tenant_id=os.getenv("AZURE_TENANT_ID")
    )
    ml_client = MLClient.from_config(credential=credential)

    @pipeline(
        description="EEG Train Pipeline with Responsible AI Dashboard",
        display_name="EEG-Train-Pipeline-RAI"
    )
    def eeg_train_pipeline():
        # Define the MLTable input
        registered_features = Input(type="mltable", path=f"azureml:automl_features:{args.version}")

        # Training component
        train_model_from_features = command(
            name="train_model_from_features",
            display_name="Train Model from Features",
            code="path/to/train_from_features.py",
            command="python train_from_features.py --registered_features ${{inputs.registered_features}} --model_output ${{outputs.model_output}}",
            environment=f"{args.environment_name}@latest",
            inputs={"registered_features": registered_features},
            outputs={"model_output": Output(type=AssetTypes.URI_FOLDER)}
        )

        # Responsible AI Dashboard
        rai_dashboard = ResponsibleAIDashboard(
            model_input=train_model_from_features.outputs.model_output,
            training_data=registered_features,
            target_column="Remission",
            task_type="classification",
            compute_name=args.compute_name
        )

        rai_dashboard.component.log_metrics(metrics=["accuracy", "f1_score", "auc"])

        return {"trained_model": train_model_from_features.outputs.model_output}

    # Configure pipeline job settings
    pipeline_job = eeg_train_pipeline()
    pipeline_job.settings.default_compute = args.compute_name
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.continue_on_step_failure = False
    pipeline_job.settings.force_rerun = True
    pipeline_job.settings.default_timeout = 3600

    # Submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )
    ml_client.jobs.stream(pipeline_job.name)

if __name__ == "__main__":
    main()
