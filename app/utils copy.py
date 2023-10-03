import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import os

# initialize mlflow url and  experiment for locally
mlflow_url = "https://mlflow.cs.ait.ac.th"
experiment_name="st124047-a3"

mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment(experiment_name)

# #Load Model
model_name = 'st124047-a3-model'
model_version = 'Staging'

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

# load the scaling parameters for both model(same scaler is used for the features for both models)
scaler_path = "scaler/scaler.pkl"
loaded_scaler_params = pickle.load(open(scaler_path, 'rb'))

# # Create scaler with the loaded parameters
loaded_scaler = StandardScaler()
loaded_scaler.mean_ = loaded_scaler_params['mean']
loaded_scaler.scale_ = loaded_scaler_params['scale']