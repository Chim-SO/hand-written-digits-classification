import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from PIL import Image

from src.models.cnn.preprocessing import scale, trim, add_border

os.environ["CUDA_VISIBLE_DEVICES"] = ""
if __name__ == '__main__':
    # Load image:
    im = Image.open("../../../data/external/test/3_0.png").convert('L')
    # im.show()

    # Trim image:
    im_trim = trim(im)
    # im_trim.show()

    # Resize image and add borders:
    b = 4
    im_res = im_trim.resize((28 - 2*b, 28 - 2*b))
    im_res = add_border(im_res, b)

    # Image preprocessing:
    x = 1 - np.array(im_res)
    x = scale(x)
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1)
    print(x.shape)

    # Load model:
    mlflow.set_tracking_uri('file:../../../mlruns')
    logged_model = 'runs:/73c37b1939b14ed2968585b761f7c9cf/models/'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict:
    y = np.argmax(loaded_model.predict(x), axis=1)[0]

    # Print:
    fig, ax = plt.subplots(1, 4)
    fig.suptitle(f'Predicted: {y}')
    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('Input')
    ax[1].imshow(im_trim, cmap='gray')
    ax[1].set_title('Trimmed')
    ax[2].imshow(im_res, cmap='gray')
    ax[2].set_title('Resized')
    ax[3].imshow(x.reshape((28, 28)), cmap='gray')
    ax[3].set_title('Preprocessed')
    plt.show()