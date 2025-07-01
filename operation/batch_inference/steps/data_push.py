from batch_inference_utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE, MODEL_NAME, USERNAME
from sagemaker.workflow.function_step import step

# Global variables
instance_type = "ml.m5.2xlarge"
image_uri = "762233743642.dkr.ecr.us-east-2.amazonaws.com/vecolazarte/batch-training@sha256:feb7cd8be10df577e2d3f9c1782563bb256eeea07060a3ea63e439830f8a4f2b"


@step(
    name="DataPush",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def data_push(experiment_name: str, run_id: str, model_inference_id: str, cod_month: str):
    
    import pandas as pd
    import mlflow
    from mlflow.artifacts import download_artifacts
    import subprocess
    subprocess.run(['pip', 'install', 'awswrangler==3.12.0']) 
    import awswrangler as wr
    import numpy as np
    from datetime import datetime
    import pytz
    import tempfile
    import os

    output_dir = tempfile.mkdtemp()

    ID_COL = "ID_CORRELATIVO"
    TIME_COL = "CODMES"
    PRED_COL = "y_prob"
    train_s3_path = f"s3://{DEFAULT_PATH}"
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    df_path = download_artifacts(run_id=model_inference_id, artifact_path=f'inf-proc-data/df_data_score_prob_{cod_month}.csv')
    df = pd.read_csv(df_path)
    
    df['attrition_profile'] = np.where(df[PRED_COL] >= 0.415, 'High risk',
                                   np.where(df[PRED_COL] >= 0.285, 'Medium risk',
                                   'Low risk'))

    df['model'] = MODEL_NAME
    timezone = pytz.timezone("America/Lima")
    df['load_date'] = datetime.now(timezone).strftime("%Y%m%d")
    df['order'] = df.y_prob.rank(method='first', ascending=False).astype(int)

    inf_posproc_s3_path = f"s3://{DEFAULT_PATH}/inf-posproc-data"
    inf_posproc_s3_path_partition = inf_posproc_s3_path + f'/output_{cod_month}.parquet'
    database = 'bank_attrition'
    table_name = database + f'.attrition_detection'

    # Pushing data to S3 path
    df = df[[ID_COL, PRED_COL, 'model','attrition_profile','load_date', 'order', TIME_COL]] 
    df.to_parquet(inf_posproc_s3_path_partition, engine='pyarrow', compression='snappy')

    # Creating table
    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (
    {ID_COL} int,
    {PRED_COL} double,
    model string,
    attrition_profile string,
    load_date string,
    order int,
    {TIME_COL} int
    )
    STORED AS parquet
    LOCATION '{inf_posproc_s3_path}'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """
    query_exec_id = wr.athena.start_query_execution(sql=ddl, database=database)
    wr.athena.wait_query(query_execution_id=query_exec_id)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="DataPush", nested=True):
            
            mlflow.log_input(mlflow.data.from_pandas(df, inf_posproc_s3_path_partition),context="DataPush")
            df.to_csv(os.path.join(output_dir, f"score_prob_{cod_month}.csv"), index=False)
            mlflow.log_artifact(os.path.join(output_dir, f"score_prob_{cod_month}.csv"), artifact_path=f"inf-posproc-data")