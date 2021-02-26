"""
This module simply provides an easy place to customise my training job. From
here I could do pretty much whatever I wanted to the model without changing
my notebook (well, within limits) 
"""

import sagemaker
from sagemaker.serializers import CSVSerializer
from config import role, sm_sess


def _init_model(output_path, model_name, rol=role):

    container = sagemaker.image_uris.retrieve('xgboost', sm_sess.boto_region_name, 'latest')

    return sagemaker.estimator.Estimator(container,
                                         rol,
                                         base_job_name=model_name,
                                         instance_count=1,
                                         instance_type='ml.m4.xlarge',
                                         output_path=output_path.as_uri(),
                                         sagemaker_session=sm_sess)


def get_xgb_model(output_path, model_name):
    xgb_model = _init_model(role, output_path, model_name)

    # Set core hyperparameters
    xgb_model.set_hyperparameters(eval_metric='merror',
                                  objective='multi:softmax',
                                  num_class=10,
                                  num_round=100)

    return xgb_model


def deploy_model(model, deploy_type='batch', batch_output=None):

    if deploy_type == 'batch':
        xgb_predictor = xgb_model.transformer(
            instance_count=1,
            instance_type='ml.m4.xlarge',
            output_path=batch_output)

    else:
        xgb_predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            serializer=CSVSerializer())

    return xgb_predictor
