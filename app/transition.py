from utils import *

def test_register_model_to_production():
    from mlflow.client import MlflowClient
    client = MlflowClient()
    for model in client.get_registered_model(model_name).latest_versions: #type: ignore
        # find model in Staging
        if(model.current_stage == 'Staging'):
            version = model.version
            client.transition_model_version_stage(
                name=model_name, version=version, stage="Production", archive_existing_versions=True
            )  
