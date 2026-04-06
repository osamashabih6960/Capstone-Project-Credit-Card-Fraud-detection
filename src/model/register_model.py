# register model
import json
import mlflow
import logging
import os
import sys
from src.logger import logging
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")



# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "osamashabih6960"
repo_name = "Capstone-Project-Credit-Card-Fraud-detection"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# -------------------------------------------------------------------------------------
# Below code is for local use
mlflow.set_tracking_uri("https://dagshub.com/osamashabih6960/Capstone-Project-Credit-Card-Fraud-detection.mlflow")
dagshub.init(repo_owner='osamashabih6960', repo_name='Capstone-Project-Credit-Card-Fraud-detection', mlflow=True)

# -------------------------------------------------------------------------------------

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model_and_transformer(model_name: str, model_info: dict, transformer_name: str, transformer_path: str):
    """Register the model and power transformer to the MLflow Model Registry."""
    try:
        client = mlflow.tracking.MlflowClient()

        # Register the model
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(model_uri)
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')

        # Log the PowerTransformer as an artifact
        mlflow.log_artifact(transformer_path, artifact_path="preprocessing")

        # Register PowerTransformer in MLflow Model Registry
        transformer_uri = f"runs:/{model_info['run_id']}/preprocessing/{os.path.basename(transformer_path)}"
        transformer_version = mlflow.register_model(transformer_uri, transformer_name)

        # Transition the PowerTransformer to "Staging"
        client.transition_model_version_stage(
            name=transformer_name,
            version=transformer_version.version,
            stage="Staging"
        )
        logging.debug(f'PowerTransformer {transformer_name} version {transformer_version.version} registered and transitioned to Staging.')

    except Exception as e:
        logging.error('Error during model and transformer registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        transformer_name = "power_transformer"
        transformer_path = "models/power_transformer.pkl"  # Ensure this file exists

        register_model_and_transformer(model_name, model_info, transformer_name, transformer_path)
    
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
