import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
import os
import json

def parse_args():
    parser = argparse.ArgumentParser("Deploy EEG Analysis Pipeline")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument("--jobtype", type=str, help="Job Type")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--sampling_rate", type=int, default=256, help="EEG Sampling Rate")
    parser.add_argument("--cutoff_frequency", type=int, default=60, help="Filter Cutoff Frequency")
    parser.add_argument("--window_seconds", type=int, default=1, help="Window Size in Seconds")
    parser.add_argument("--version", type=str, help="Version of registered features")
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
    
    # Data loading component
    data_loader = command(
        name="data_loader",
        display_name="load-data",
        code=os.path.join(parent_dir, args.jobtype),
        command="python data_loader.py \
                --input_data ${{inputs.input_data}} \
                --output_data ${{outputs.output_data}}",
        environment=args.environment_name+"@latest",
        inputs={
            "input_data": Input(type="uri_file")
        },
        outputs={
            "output_data": Output(type="uri_folder")
        }
    )

    # Upsampling component
    upsampler = command(
        name="upsampler",
        display_name="upsample-data",
        code=os.path.join(parent_dir, args.jobtype),
        command="python upsampler.py \
                --input_data ${{inputs.input_data}} \
                --output_data ${{outputs.output_data}} \
                --upsampling_factor 2",
        environment=args.environment_name+"@latest",
        inputs={
            "input_data": Input(type="uri_folder")
        },
        outputs={
            "output_data": Output(type="uri_folder")
        }
    )

    # Filtering component
    filter_data = command(
        name="filter",
        display_name="filter-data",
        code=os.path.join(parent_dir, args.jobtype),
        command="python filter.py \
                --input_data ${{inputs.input_data}} \
                --output_data ${{outputs.output_data}} \
                --sampling_rate ${{inputs.sampling_rate}} \
                --cutoff_frequency ${{inputs.cutoff_frequency}}",
        environment=args.environment_name+"@latest",
        inputs={
            "input_data": Input(type="uri_folder"),
            "sampling_rate": Input(type="number"),
            "cutoff_frequency": Input(type="number")
        },
        outputs={
            "output_data": Output(type="uri_folder")
        }
    )

    # Downsampling component
    downsampler = command(
        name="downsampler",
        display_name="downsample-data",
        code=os.path.join(parent_dir, args.jobtype),
        command="python downsampler.py \
                --input_data ${{inputs.input_data}} \
                --output_data ${{outputs.output_data}}",
        environment=args.environment_name+"@latest",
        inputs={
            "input_data": Input(type="uri_folder")
        },
        outputs={
            "output_data": Output(type="uri_folder")
        }
    )

    window_slicer = command(
        name="window_slicer",
        display_name="window-slicer",
        code=os.path.join(parent_dir, args.jobtype),
        command="python window_slicer.py \
                --input_data ${{inputs.input_data}} \
                --output_data ${{outputs.output_data}} \
                --window_seconds ${{inputs.window_seconds}} \
                --sampling_rate ${{inputs.sampling_rate}}",
        environment=args.environment_name+"@latest",
        inputs={
            "input_data": Input(type="uri_folder"),
            "sampling_rate": Input(type="number"),
            "window_seconds": Input(type="number")
        },
        outputs={
            "output_data": Output(type="uri_folder")
        }
    )

    # Feature extraction component
    extract_features = command(
        name="extract_features",
        display_name="extract-features",
        code=os.path.join(parent_dir, args.jobtype),
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

    # Pass the JSON string to the register_features command
    register_features = command(
        name="register_features",
        display_name="register-features",
        code=os.path.join(parent_dir, args.jobtype),
        command="python register_features.py \
                --features_input ${{inputs.features_input}} \
                --data_name ${{inputs.data_name}} \
                --registered_features_output ${{outputs.registered_features}} \
                --subscription_id ${{inputs.subscription_id}} \
                --resource_group ${{inputs.resource_group}} \
                --workspace_name ${{inputs.workspace_name}} \
                --client_id ${{inputs.client_id}} \
                --client_secret ${{inputs.client_secret}} \
                --tenant_id ${{inputs.tenant_id}} \
                --version ${{inputs.version}}",
        environment=args.environment_name+"@latest",
        inputs={
            "features_input": Input(type="uri_folder"),
            "data_name": Input(type="string"),
            "subscription_id": Input(type="string"),
            "resource_group": Input(type="string"),
            "workspace_name": Input(type="string"),
            "client_id": Input(type="string"),
            "client_secret": Input(type="string"),
            "tenant_id": Input(type="string"),
            "version": Input(type="string")
        },
        outputs={
            "registered_features": Output(type="uri_file")
        }
    )


    @pipeline(
        description="EEG Analysis Pipeline for Depression Classification",
        display_name="EEG-Analysis-Pipeline"
    )
    def eeg_analysis_pipeline(raw_data, sampling_rate, cutoff_frequency, feature_data_name, window_seconds, version):
        # Load data
        load = data_loader(
            input_data=raw_data
        )

        # Upsample data
        upsampled = upsampler(
            input_data=load.outputs.output_data
        )

        # Apply filtering
        filtered = filter_data(
            input_data=upsampled.outputs.output_data,
            sampling_rate=sampling_rate,
            cutoff_frequency=cutoff_frequency
        )

        # Downsample filtered data
        downsampled = downsampler(
            input_data=filtered.outputs.output_data
        )

        windowed = window_slicer(
            input_data=downsampled.outputs.output_data,
            sampling_rate=sampling_rate,
            window_seconds=window_seconds
        )
        # Extract features
        features = extract_features(
            processed_data=windowed.outputs.output_data,
            sampling_rate=sampling_rate
        )

        registered = register_features(
            features_input=features.outputs.features_output,
            data_name=feature_data_name,
            subscription_id=ml_client.subscription_id,
            resource_group=ml_client.resource_group_name,
            workspace_name=ml_client.workspace_name,
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"],
            version=version
        )

        return {
            "loaded_data": load.outputs.output_data,
            "upsampled_data": upsampled.outputs.output_data,
            "filtered_data": filtered.outputs.output_data,
            "downsampled_data": downsampled.outputs.output_data,
            "windowed_data": windowed.outputs.output_data,
            "features": features.outputs.features_output,
            "registered_features": registered.outputs.registered_features
        }

    # Create pipeline job
    pipeline_job = eeg_analysis_pipeline(
        Input(path=args.data_name + "@latest", type="uri_file"),
        args.sampling_rate,
        args.cutoff_frequency,
        "eeg_features",
        args.window_seconds,
        args.version
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
