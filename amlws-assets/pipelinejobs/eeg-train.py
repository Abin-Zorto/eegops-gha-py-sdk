import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
import os

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--sampling_rate", type=int, default=256, help="EEG Sampling Rate")
    parser.add_argument("--cutoff_frequency", type=int, default=60, help="Filter Cutoff Frequency")
    args = parser.parse_args()
    return args

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
    
    prep_data = command(
        name="prep_data",
        display_name="prep-data",
        code=os.path.join(parent_dir),
        command="python prep.py \
                --input_data ${{inputs.input_data}} \
                --processed_data ${{outputs.processed_data}} \
                --sampling_rate ${{inputs.sampling_rate}} \
                --cutoff_frequency ${{inputs.cutoff_frequency}}",
        environment=args.environment_name+"@latest",
        inputs={
            "input_data": Input(type="uri_file"),
            "sampling_rate": Input(type="number"),
            "cutoff_frequency": Input(type="number")
        },
        outputs={
            "processed_data": Output(type="uri_folder")
        }
    )

    extract_features = command(
        name="extract_features",
        display_name="extract-features",
        code=os.path.join(parent_dir),
        command="python extract_features.py \
                --processed_data ${{inputs.processed_data}} \
                --features_output ${{outputs.features_output}} \
                --sampling_rate ${{inputs.sampling_rate}}",
        environment=args.environment_name+"@latest",
        inputs={
            "processed_data": Input(type="uri_folder"),
            "sampling_rate": Input(type="number")
        },
        outputs={
            "features_output": Output(type="uri_folder")
        }
    )

    train_model = command(
        name="train_model",
        display_name="train-model",
        code=os.path.join(parent_dir),
        command="python train.py \
                --features_input ${{inputs.features_input}} \
                --model_output ${{outputs.model_output}}",
        environment=args.environment_name+"@latest",
        inputs={
            "features_input": Input(type="uri_folder")
        },
        outputs={
            "model_output": Output(type="uri_folder")
        }
    )

    @pipeline()
    def eeg_analysis_pipeline(raw_data, sampling_rate, cutoff_frequency):
        prep = prep_data(
            input_data=raw_data,
            sampling_rate=sampling_rate,
            cutoff_frequency=cutoff_frequency
        )

        features = extract_features(
            processed_data=prep.outputs.processed_data,
            sampling_rate=sampling_rate
        )

        model = train_model(
            features_input=features.outputs.features_output
        )

        return {
            "processed_data": prep.outputs.processed_data,
            "features": features.outputs.features_output,
            "trained_model": model.outputs.model_output
        }

    # Create pipeline job
    pipeline_job = eeg_analysis_pipeline(
        Input(path=args.data_name + "@latest", type="uri_file"),
        args.sampling_rate,
        args.cutoff_frequency
    )

    # Set pipeline level compute
    pipeline_job.settings.default_compute = args.compute_name
    # Set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    # Submit and monitor pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )
    ml_client.jobs.stream(pipeline_job.name)

if __name__ == "__main__":
    main()