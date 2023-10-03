from utils import *

register_model_to_production()
loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_stage}")