FROM python:3.9

# Actualizar pip
RUN pip install --upgrade pip

# Instalar librerías requeridas
RUN pip install --no-cache-dir \
    mlflow==2.13.2 \
    sagemaker==2.244.0 \
    sagemaker-mlflow==0.1.0 \
    xgboost==2.1.4 \
    boto3 \
    numpy \
    pandas \
    awswrangler==3.4.0

# Validación opcional de instalación
RUN python -c "import mlflow, xgboost, boto3, numpy, pandas; print('Librerías instaladas correctamente')"
