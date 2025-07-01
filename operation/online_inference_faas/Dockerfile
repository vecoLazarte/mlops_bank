# Use an AWS Lambda-compatible base image
FROM public.ecr.aws/lambda/python:3.9
#FROM public.ecr.aws/sagemaker/sagemaker-distribution:3.1-cpu

# Install dependencies
# Instalar librerías requeridas
RUN pip install --no-cache-dir \
    mlflow==2.22 \
    sagemaker==2.244.0 \
    sagemaker-mlflow==0.1.0 \
    xgboost==2.1.4 \
    boto3 \
    numpy \
    pandas \
    awswrangler==3.12.0\
    fsspec==2024.12.0\
    s3fs==2024.12.0


# Copy entrypoint script
COPY entrypoint.py /var/task/entrypoint.py

# Set entrypoint
CMD ["entrypoint.lambda_handler"]