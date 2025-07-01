# Imagen base de SageMaker con Python
FROM python:3.10-slim

# Instala dependencias necesarias
RUN pip install --no-cache-dir \
    mlflow==2.13.2 \
    sagemaker-mlflow==0.1.0 \
    xgboost==2.1.4 \
    boto3 \
    numpy \
    pandas \
    awswrangler==3.12.0

# Define el working dir (opcional)
WORKDIR /opt/ml/code
