from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser("register_environment")
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--description", type=str)
    parser.add_argument("--env_path", type=str)
    parser.add_argument("--build_type", type=str)
    parser.add_argument("--base_image", type=str)
    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Read YAML file
    with open("config.json") as f:
        config = yaml.safe_load(f)

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential
    )

    # Read the conda file
    with open(args.env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    env = Environment(
        name=args.environment_name,
        description=args.description,
        conda_file=conda_env,
        image=args.base_image
    )

    ml_client.environments.create_or_update(env)

    print(f"Environment {args.environment_name} registered successfully")

if __name__ == "__main__":
    main()