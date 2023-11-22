import argparse
import itertools

import mlflow
import yaml

from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning of a Keras CNN-based model for MNIST classification")
    parser.add_argument("--config-file", "-c", type=str, default='configs/cnnbased.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters:
    data_path = config['data']['dataset_path']

    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['mlruns_path'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Compute all possible combinations:
    params = config['hyperparameter_tuning']
    keys, values = zip(*params.items())
    runs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Train:
    for run in runs:
        with mlflow.start_run():
            train(data_path, run['num_epochs'], run['batch_size'])