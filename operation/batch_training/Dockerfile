FROM python:3.12

# Desactiva buffering para logs
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema necesarias para compilar pyarrow
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python
RUN pip install --upgrade pip && pip install --no-cache-dir \
    mlflow==2.13.2 \
    sagemaker-mlflow==0.1.0 \
    xgboost==2.1.4 \
    boto3 \
    numpy \
    pandas \
    awswrangler==2.20.0
