from batch_training_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE, MODEL_NAME
from sagemaker.workflow.function_step import step

instance_type = "ml.m5.2xlarge"
default_path = DEFAULT_PATH
image_uri = "762233743642.dkr.ecr.us-east-2.amazonaws.com/vecolazarte/batch-training:latest@sha256:8301d35f2fff625f01304ef65977b41a074860fa441b1ac19a266595fa9bac27"

@step(
    name="RegisterXGBoostModel", 
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def register_xgboost_model(experiment_name: str, name_path: str, run_id: str, evaluation_run_id: str):
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'awswrangler==3.12.0', 'sagemaker==2.244.0'])
    import mlflow
    from mlflow.artifacts import download_artifacts
    from mlflow.models.signature import infer_signature
    from mlflow.tracking import MlflowClient
    import os
    import pickle
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    import tempfile
    output_dir = tempfile.mkdtemp()
    
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)
    
    train_s3_path = f"s3://{default_path}"
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="RegisterXGBoostModel", nested=True):
            
            model_uri = f"runs:/{evaluation_run_id}/{name_path}_model"
            model_registry_name = MODEL_NAME
            result = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
            client = MlflowClient()
        
            client.set_model_version_tag(
            name=model_registry_name,
            version=result.version,
            key="estado",
            value="production")

            client.set_registered_model_alias(
            name=model_registry_name,
            alias="champion",
            version=result.version)

            client.update_model_version(
            name=model_registry_name,
            version=result.version,
            description=f"{name_path} fue el modelo que obtuvo mejor recall usando los mejores hiperparametros por lo que ahora sera el modelo productivo")