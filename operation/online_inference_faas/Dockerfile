FROM public.ecr.aws/lambda/python:3.9

# 1. Instala dependencias del sistema necesarias para compilar Pillow
RUN yum install -y \
    gcc \
    zlib-devel \
    libjpeg-devel \
    libpng-devel \
    freetype-devel \
    && yum clean all

# 2. Instala las librerías Python requeridas
RUN pip install --no-cache-dir \
    mlflow==2.22 \
    sagemaker==2.244.0 \
    sagemaker-mlflow==0.1.0 \
    xgboost==2.1.4 \
    boto3 \
    numpy \
    pandas \
    awswrangler==3.12.0 \
    fsspec==2024.12.0 \
    s3fs==2024.12.0

# 3. Copiar tu entrypoint
COPY entrypoint.py /var/task/entrypoint.py

# 4. Configurar el handler para Lambda
CMD ["entrypoint.lambda_handler"]
