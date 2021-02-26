from time import sleep

import pandas as pd
import smart_open as smart

from config import boto_session, sm_client


def get_initial_predictions(tuner, input_data, output_path, model_save_name):
    """
    Used on the test data to assess error. Also handles the initial deploy so we
    can run further batch transforms.

    Parameters
    ----------
    tuner : sagemaker.tuner
        Model tuning job
    input_data : s3fs.PureS3Path
        Where are the input data on S3?
    output_path : s3fs.PureS3Path
        Where to save the outputs on S3
    model_save_name : str
        Name to save model under

    Returns
    -------
    pd.DataFrame
        Of predictions, single columned (called 0) in this instance
    """

    best_model = tuner.best_estimator()
    batch_job = best_model.transformer(1, "ml.m5.large", output_path=output_path.as_uri(),
                                       model_name=model_save_name)
    batch_job.transform(input_data.as_uri())
    # TODO: Do an ls first so we can get any/all files
    output_file = output_path / 'validation.csv.out'
    with smart.open(output_file.as_uri(), 'r', transport_params={'session': boto_session}) as f:
        predictions = pd.read_csv(f, header=None)
    return predictions


def get_subsequent_predictions(batch_job_name, model_name, input_path, output_path):
    """
    Once a model has been deployed, this can be used to run it in the future.
    """
    request = {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "TransformOutput": {
            "S3OutputPath": output_path.as_uri(),
            "Accept": "text/csv",
            "AssembleWith": "Line"
        },
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_path.as_uri()
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
            "CompressionType": "None"
        },
        "TransformResources": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1
        }
    }
    sm_client.create_transform_job(**request)

    status = "InProgress"
    while status == "InProgress":
        sleep(15)
        status = sm_client.describe_transform_job(TransformJobName=batch_job_name)['TransformJobStatus']
        print(status)
    if status != "Completed":
        raise RuntimeError("Batch Inference Not Completed Successfully!")

    # TODO: Do an ls first so we can get any/all files
    output_file = output_path / 'predict.csv.out'
    with smart.open(output_file.as_uri(), 'r', transport_params={'session': boto_session}) as f:
        predictions = pd.read_csv(f, header=None)
    return predictions


def deploy_api(tuner, model_name):
    # Deploy from tuner to API, returned predictor contains api access
    return tuner.deploy(1, "ml.t2.medium", model_name=model_name)

    # When done
    # predictor.delete_endpoint()