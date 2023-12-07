import argparse
import os

import yaml

from src.models.cnn.dataloader import load_data


def check_image_dimensions(dataset, dimension):
    assert dataset.shape[1] == (dimension[0] * dimension[1]), f"Image dimensions do not match the expected shape : {dimension[0] * dimension[1]}"
    print(f"Checking images dimensions : {dimension[0] * dimension[1]} ... OK!")


def check_y_dimensions(y, dimension):
    assert y.ndim == dimension, f"Labels dimensions do not match the expected dimension: {dimension}"
    print(f"Checking labels dimensions : {dimension} ... OK!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Testing ...")
    parser.add_argument("--config-file", "-c", type=str, default='configs/cnn.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters:
    data_path = config['data']['dataset_path']
    image_dimension = config['data']['image_dimension']
    y_dim = config['data']['label_dimension']

    # Load data:
    x_train, y_train = load_data(os.path.join(data_path, 'train.csv'))
    x_test, y_test = load_data(os.path.join(data_path, 'test.csv'))

    # Execute tests:
    check_image_dimensions(x_train, image_dimension)
    check_image_dimensions(x_test, image_dimension)
    check_y_dimensions(y_train, y_dim)
    check_y_dimensions(y_test, y_dim)