import sagemaker
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir)))
from utils import DEFAULT_BUCKET, ENV_CODE, TRACKING_SERVER_ARN, USERNAME, ENV_CODE

# Sagemaker configuration
SAGEMAKER_ROLE = "arn:aws:iam::762233743642:role/service-role/AmazonSageMaker-ExecutionRole-20250624T115826"
default_prefix = f"sagemaker/bank-attrition-detection"
DEFAULT_PATH = DEFAULT_BUCKET + "/" + default_prefix
sagemaker_session = sagemaker.Session(default_bucket=DEFAULT_BUCKET,
                                      default_bucket_prefix=default_prefix)
#Pipeline configuration
PIPELINE_NAME = f"pipeline-inference-{ENV_CODE}-{USERNAME}"
MODEL_NAME = f"attrition-detection-{USERNAME}"
MODEL_VERSION = "latest"
TRACKING_SERVER_ARN = TRACKING_SERVER_ARN
USERNAME = USERNAME
ENV_CODE = ENV_CODE

