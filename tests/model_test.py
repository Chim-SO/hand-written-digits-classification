import argparse
import os

import mlflow.pyfunc
import numpy as np
import yaml
from sklearn.metrics import f1_score

from src.models.cnn.dataloader import load_data
from src.models.cnn.preprocessing import reshape, scale


def test_model_specification(model, input_dim, output_dim):
    inputs = model.metadata.signature.inputs.to_dict()
    outputs = model.metadata.signature.outputs.to_dict()
    assert inputs[0]['tensor-spec']['shape'][
           -3:] == input_dim, f"Model input dimension does not match the expected shape : {input_dim}"
    print(f"Checking model input dimension : {input_dim} ... OK!")
    assert outputs[0]['tensor-spec']['shape'][
               -1] == output_dim, f"Model output dimension does not match the expected shape : {output_dim}"
    print(f"Checking model output dimension : {output_dim} ... OK!")


def test_model_validation(model, x, y):
    f1 = float(f1_score(y, np.argmax(model.predict(x), axis=1), average='macro'))
    assert f1 >= 0.95, f"F1 score is less than 0.95: {f1}"
    print(f"Checking F1 score : {f1} ... OK!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Testing ...")
    parser.add_argument("--config-file", "-c", type=str, default='configs/cnn.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters:
    data_path = config['data']['dataset_path']
    input_dim = config['model']['input_dim']
    y_dim = config['model']['output_dim']
    x_test, y_test = load_data(os.path.join(data_path, 'test.csv'))
    # Reshape:
    x_test = reshape(x_test)
    # Scaling:
    x_test = scale(x_test)

    # Load model as a PyFuncModel:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    loaded_model = mlflow.pyfunc.load_model(
        'mlflow-artifacts:/673114994354686159/6e143279eae147bdb90874389d4c2a24/artifacts/models')

    # test model:
    test_model_specification(loaded_model, tuple(input_dim), y_dim)
    test_model_validation(loaded_model, x_test, y_test)
