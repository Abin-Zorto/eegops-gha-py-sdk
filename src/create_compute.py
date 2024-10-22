"""MLOps v2 NLP Python SDK register environment script."""
import os
import argparse
import traceback

# Azure ML sdk v2 imports
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.core.exceptions import ResourceExistsError
from azure.ai.ml.entities import AmlCompute
from azure.identity import ClientSecretCredential

def get_config_parger(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for the script."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--subscription_id",
        type=str,
        required=False,
        help="Subscription ID",
    )
    parser.add_argument(
        "--resource_group",
        type=str,
        required=False,
        help="Resource group name",
    )
    parser.add_argument(
        "--workspace_name",
        type=str,
        required=False,
        help="Workspace name",
    )
    parser.add_argument(
        "--cluster_name",
        type=str,
        required=False,
        help="Name of Cluster to create",
    )
    parser.add_argument(
        "--size",
        type=str,
        required=False,
        help="Size of VM to be created",
    )
    parser.add_argument(
        "--min_instances",
        type=str,
        required=False,
        help="Min number of instances",
    )
    parser.add_argument(
        "--max_instances",
        type=str,
        required=False,
        help="Max number of instances",
    )
    parser.add_argument(
        "--cluster_tier",
        type=str,
        required=False,
        help="Cluster tier",
    )
    return parser


def connect_to_aml(args):
    """Connect to Azure ML workspace using provided cli arguments."""
    credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"]
        )
    ml_client = MLClient.from_config(credential=credential)
    return ml_client


def main():
    """Main entry point for the script."""
    parser = get_config_parger()
    args, _ = parser.parse_known_args()
    ml_client = connect_to_aml(args)

    try:
        # Check if the cluster already exists
        cluster = ml_client.compute.get(args.cluster_name)
        print(f"Cluster '{args.cluster_name}' already exists. Skipping creation.")
    except Exception:
        print(f"Cluster '{args.cluster_name}' not found. Creating new cluster.")
        
        cluster_basic = AmlCompute(
            name=args.cluster_name,
            type="amlcompute",
            size=args.size,
            min_instances=args.min_instances,
            max_instances=args.max_instances,
            tier=args.cluster_tier,
        )
        ml_client.begin_create_or_update(cluster_basic).result()


if __name__ == "__main__":
    main()