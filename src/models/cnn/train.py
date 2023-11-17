import argparse
import os.path
import random

import numpy as np
import tensorflow as tf
import yaml
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
    acc = round(accuracy_score(actual, pred, normalize=True) * 100, 2)
    precision = round(precision_score(actual, pred, average='macro') * 100, 2)
    recall = round(recall_score(actual, pred, average='macro') * 100, 2)
    f1 = round(f1_score(actual, pred, average='macro') * 100, 2)
    return acc, precision, recall, f1


def train(data_path, epochs, batch_size):
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
    metric = 'accuracy'
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
          config['training']['batch_size'])
