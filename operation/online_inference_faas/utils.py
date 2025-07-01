import os

DEFAULT_BUCKET = "mlops-chester"
ENV_CODE = "prod"
TRACKING_SERVER_ARN = 'arn:aws:sagemaker:us-east-2:762233743642:mlflow-tracking-server/mlops-mlflow-server'
USERNAME = os.getenv("GITHUB_ACTOR")



