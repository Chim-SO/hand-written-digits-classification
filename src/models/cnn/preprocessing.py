import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageOps


def scale(data, factor=255):
    data = data.astype("float32") / factor
    return data


def reshape(data):
    data_shape = data.shape
    if data_shape == (28, 28):
        data = np.expand_dims(data, 0)
        data = np.expand_dims(data, -1)
    elif data_shape == (28, 28, 1):
        data = np.expand_dims(data, 0)
    else:
        data = data.reshape((data.shape[0], 28, 28, 1))
    return data


def onehot_encoding(feature, num_cat=10):
    return tf.keras.utils.to_categorical(feature, num_classes=num_cat)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox)


def add_border(im, border=5, fill=255):
    return ImageOps.expand(im, border=border, fill=fill)
