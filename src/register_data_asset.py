from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import ClientSecretCredential
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="Name of the data asset")
    parser.add_argument("--description", type=str, help="Description of the data asset")
    parser.add_argument("--data_type", type=str, help="Type of the data asset")
    parser.add_argument("--data_path", type=str, help="Path to the data asset")
    args = parser.parse_args()

    try:
        # Use ClientSecretCredential instead of DefaultAzureCredential
        credential = ClientSecretCredential(
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
            tenant_id=os.environ["AZURE_TENANT_ID"]
        )
        
        # Include subscription_id when initializing MLClient
        ml_client = MLClient.from_config(
            credential=credential
        )

        print(f"Registering data asset {args.data_name}")
        
        data = Data(
            name=args.data_name,
            description=args.description,
            path=args.data_path,
            type=args.data_type
        )

        ml_client.data.create_or_update(data)
        print(f"Data asset {args.data_name} registered successfully")

    except Exception as ex:
        print(f"An error occurred: {ex}")
        raise

if __name__ == "__main__":
    main()