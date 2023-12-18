import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from PIL import Image

from src.models.cnn.preprocessing import scale, trim, add_border, reshape

os.environ["CUDA_VISIBLE_DEVICES"] = ""
if __name__ == '__main__':
    # Load image:
    im = Image.open("../../../data/external/tests/7_0.png").convert('L')
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
    x = reshape(x)
    print(x.shape)

    # Load model:
    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    # Search for runs in the experiment and get the latest run
    runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name('cnn').experiment_id], order_by=["start_time desc"])

    if not runs.empty:
        latest_run_id = runs.iloc[0]["run_id"]
        # Load the model from the latest run
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/models")

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
        plt.savefig('sample_plot.png', bbox_inches='tight')
        plt.show()
    else:
        print("No runs found in the specified experiment.")