import boto3
import pytest
from botocore.exceptions import ClientError

def s3_path_exists(bucket_name, key):
    """Checks if an S3 object (path) exists."""
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

@pytest.mark.parametrize("bucket_name, key, expected", [
    ("mlops-chester", "mlflow-server/67/70e54723a106442f9b58d23accb1d18a/artifacts/xgbost_model/model.xgb", True),  
    ("mlops-chester", "mlflow-server/67/70e54723a106442f9b58d23accb1d18a/artifacts/random_forest_model/model.pkl", True) 
])
def test_s3_path(bucket_name, key, expected):
    """Test if an S3 path exists or not."""
    assert s3_path_exists(bucket_name, key) == expected, f"Path '{key}' existence does not match expected value!"

