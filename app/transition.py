from utils import *


target_version = "Production"

# List available versions of the model
model_versions = mlflow.search_model_versions(f"name='{model_name}'")

# Find the version to transition and update
transition_version = None
for _, row in model_versions.iterrows():
    if row["version"] == model_version:
        transition_version = row
        break

if transition_version is None:
    print(f"Model version {model_version} not found.")
else:
    # Transition the model to the desired stage
    mlflow.transition_model_version_stage(
        name=model_name,
        version=transition_version["version"],
        stage=target_version
    )

    print(f"Model version {model_version} has been transitioned to '{target_version}' stage.")
    loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{target_version}")