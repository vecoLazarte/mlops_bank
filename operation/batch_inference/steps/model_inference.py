from batch_inference_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE, MODEL_NAME, MODEL_VERSION, default_prefix, DEFAULT_BUCKET
from sagemaker.workflow.function_step import step

# Global variables
instance_type = "ml.m5.2xlarge"
image_uri = "762233743642.dkr.ecr.us-east-2.amazonaws.com/vecolazarte/batch-training@sha256:feb7cd8be10df577e2d3f9c1782563bb256eeea07060a3ea63e439830f8a4f2b"

@step(
    name="ModelInference",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def model_inference(experiment_name: str, run_id: str, data_pull_id: str, cod_month: int) -> str:
    import os
    import pandas as pd
    import mlflow
    from mlflow.artifacts import download_artifacts
    import boto3
    import tempfile

    output_dir = tempfile.mkdtemp()
    s3 = boto3.client("s3")

    train_s3_path = f"s3://{DEFAULT_PATH}"

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    flavors = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{MODEL_VERSION}").flavors

    if "xgboost" in flavors:
        import mlflow.xgboost
        model = mlflow.xgboost.load_model(model_uri)
    elif "sklearn" in flavors:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(model_uri)
    
    info = mlflow.models.get_model_info(model_uri)
    artifact_path = info.artifact_path
    name_model = artifact_path.replace('_model', '')
    
    df_data_score_path = download_artifacts(run_id=data_pull_id, artifact_path=f"inf-raw-data/df_data_score_prepared_{cod_month}.csv")
    df_data_score = pd.read_csv(df_data_score_path)

    s3_key = f'{default_prefix}/outputs/train/feature_importance/{name_model}/feature_importance.csv'
    local_path = 'feature_importance.csv'
    s3.download_file(DEFAULT_BUCKET, s3_key, local_path)
    features = pd.read_csv(local_path)['variable'].to_list()

    y_pred = model.predict_proba(df_data_score[features])
    df_data_score['y_prob'] = y_pred[:,1]

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelInference", nested=True) as model_inference:
            model_inference_id = model_inference.info.run_id
            df_data_score.to_csv(os.path.join(train_s3_path, "inf-proc-data", f"df_data_score_prob_{cod_month}.csv"), index=False)
            df_data_score.to_csv(os.path.join(output_dir, f"df_data_score_prob_{cod_month}.csv"), index=False)
            mlflow.log_artifact(os.path.join(output_dir, f"df_data_score_prob_{cod_month}.csv"), artifact_path=f"inf-proc-data")
            mlflow.log_input(mlflow.data.from_pandas(df_data_score, os.path.join(train_s3_path, "inf-proc-data", f"df_data_score_prob_{cod_month}.csv")), context="ModelInference")
    return model_inference_id