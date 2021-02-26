
from pathlib import Path
import os

from s3path import PureS3Path
import boto3
from botocore.exceptions import ClientError
import sagemaker

# Main names and paths
model_name = "miguel-learning-xgboost-4"
data_dir = Path("YOUR_LOCAL_DATA_DIR")
s3_root = PureS3Path('/miguel-sagemaker/')

# data dirs in S3
s3_data_loc = s3_root / "data"
s3_model_loc = s3_root / "model"

# batch output
s3_predict_out = s3_data_loc / 'predictions'

# Boto3 connection (this handles talking to AWS)
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
boto_session = boto3.session.Session(profile_name='saml')

# sagemaker specific AWS interfaces
sm_sess = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client('sagemaker')

# role to define my permissions on AWS, autopopulates if you are on sagemaker.
try:
    role = sagemaker.get_execution_role()
except ClientError:
    role = "" # Pulled from IAM roles
