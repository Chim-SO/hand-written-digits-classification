import argparse
import os

import mlflow.keras
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from matplotlib import pyplot as plt

from src.models.cnn.dataloader import load_data
from src.models.cnn.preprocessing import reshape, scale


def get_grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the conv layer
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class:
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron (top predicted or chosen)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # normalize the heatmap:
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def superimpose_heatmap(original_img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap, 'L')
    heatmap = heatmap.resize(original_img.size)
    heatmap = heatmap.convert("RGB")
    original_img = original_img.convert("RGB")
    # Superimpose the heatmap on the original image
    superimposed_img = Image.blend(original_img, heatmap, alpha=0.5)
    return superimposed_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GRAD-CAM: Gradient-weighted Class Activation Mapping")
    parser.add_argument("--config-file", "-c", type=str, default='../configs/cnn.yaml')
    args = parser.parse_args()

    # Load model as a PyFuncModel:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    model = mlflow.tensorflow.load_model(
        'mlflow-artifacts:/673114994354686159/6e143279eae147bdb90874389d4c2a24/artifacts/models')
    print(model.summary())

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
    x_test_processed = scale(x_test)

    # Get the Grad-CAM heatmap
    ind = 10
    img_array = np.expand_dims(x_test_processed[ind], 0)
    print(img_array.shape)
    heatmap = get_grad_cam(model, img_array, 'conv2d_1')

    # Superimpose the heatmap on the original image
    original_img = Image.fromarray(np.squeeze(x_test[ind]).astype(np.uint8), mode='L')
    result_img = superimpose_heatmap(original_img, heatmap)

    # Display the original image, heatmap, and superimposed image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='viridis')
    plt.title('Grad-CAM Heatmap')

    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title('Superimposed Image')

    plt.show()
