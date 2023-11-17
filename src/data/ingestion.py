import argparse
import os.path
import urllib.request


def download_raw_data(raw_data_path):
    train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    # Download the MNIST dataset
    urllib.request.urlretrieve(train_images_url, os.path.join(raw_data_path, 'train_images.gz'))
    urllib.request.urlretrieve(train_labels_url, os.path.join(raw_data_path, 'train_labels.gz'))
    urllib.request.urlretrieve(test_images_url, os.path.join(raw_data_path, 'test_images.gz'))
    urllib.request.urlretrieve(test_labels_url, os.path.join(raw_data_path, 'test_labels.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("-r", "--raw-path", help="Raw data path", required=True)
    args = parser.parse_args()

    download_raw_data(args.raw_path)
