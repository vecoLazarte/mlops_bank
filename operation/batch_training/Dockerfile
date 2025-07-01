FROM python:3.10-slim

# Actualizar pip
RUN pip install --upgrade pip

# Instalar mlflow primero (usa pyarrow<16)
RUN pip install mlflow==2.13.2

# Luego instalar awswrangler sin pyarrow
RUN pip install awswrangler==3.12.0 --no-deps

# Finalmente instalar sagemaker
RUN pip install sagemaker==2.244.0
