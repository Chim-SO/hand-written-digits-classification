import argparse
import os.path
import random

import mlflow.keras
import numpy as np
import tensorflow as tf
import yaml
from mlflow.models.signature import infer_signature
from numpy.random import seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.models.cnn.dataloader import load_data
from src.models.cnn.model import create_model
from src.models.cnn.preprocessing import scale, reshape, onehot_encoding

seed(1)
tf.random.set_seed(1)
tf.config.experimental.enable_op_determinism()
random.seed(2)


def eval_metrics(actual, pred):
    return (float(accuracy_score(actual, pred, normalize=True)),
            float(precision_score(actual, pred, average='macro')),
            float(recall_score(actual, pred, average='macro')),
            float(f1_score(actual, pred, average='macro')))


def train(data_path, epochs, batch_size, output_path, tracking_uri, experiment_name, tags):
    # Read dataset:
    x_train, y_train = load_data(os.path.join(data_path, 'train.csv'))
    # Scaling:
    x_train = scale(x_train)
    # Reshape:
    x_train = reshape(x_train)
    # Encode the output:
    y_train = onehot_encoding(y_train)
    # Split dataset:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    # Create model:
    model = create_model(x_train[0].shape)

    # Log parameters:
    loss = 'categorical_crossentropy'
    metric = 'accuracy'

    # Train:
    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(x_val, y_val))

    # Evaluation
    x_test, y_test = load_data(os.path.join(data_path, 'test.csv'))
    # Reshape:
    x_test = reshape(x_test)
    # Scaling:
    x_test = scale(x_test)
    # Encode the output:
    y_test = onehot_encoding(y_test)

    # evaluate and log:
    test_loss, test_metric = model.evaluate(x_test, y_test, verbose=2)

    # Other metric evaluation:
    y_pred = np.argmax(model.predict(x_test), axis=1)
    acc, precision, recall, f1 = eval_metrics(np.argmax(y_test, axis=1), y_pred)


    # Set tracking server uri for logging
    mlflow.set_tracking_uri(tracking_uri)
    # Create an MLflow Experiment
    mlflow.set_experiment(experiment_name)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params({
            'loss': loss,
            'metric': metric,
            'epochs': epochs,
            'batch_size': batch_size
        })
        # Log the metrics
        mlflow.log_metrics({
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_loss': history.history['loss'][-1],
            'training_acc': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_acc': history.history['val_accuracy'][-1],
            'test_loss': test_loss,
            'test_metric': test_metric
        })
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tags(tags)

        # Save model:
        signature = infer_signature(x_train, y_train)
        mlflow.tensorflow.log_model(model, output_path, signature=signature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Keras CNN-based model for MNIST classification")
    parser.add_argument("--config-file", "-c", type=str, default='configs/cnnbased.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters:
    data_path = config['data']['dataset_path']

    train(data_path, config['training']['num_epochs'],
          config['training']['batch_size'], config['training']['output_path'], config['mlflow']['tracking_uri'], config['mlflow']['experiment_name'], config['mlflow']['tags'])
