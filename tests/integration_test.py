import os
import subprocess
import unittest

import mlflow
import numpy as np
import yaml
from sklearn.metrics import f1_score

from src.models.cnn.dataloader import load_data
from src.models.cnn.preprocessing import reshape, scale


class TestHandwrittenDigitsPipelineIntegration(unittest.TestCase):

    def setUp(self):
        # Load the configuration file:
        with open('configs/cnn.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Get parameters:
        data_path = config['data']['dataset_path']
        self.x_test, self.y_test = load_data(os.path.join(data_path, 'test.csv'))
        # Reshape:
        self.x_test = reshape(self.x_test)
        # Scaling:
        self.x_test = scale(self.x_test)

    def test_train_integration(self):
        # Train the model:
        subprocess.run(["python", "-m", "src.models.cnn.train", "c", "configs/cnn.yaml"])

        # Make predictions of the latest model:
        subprocess.run(["python", "-m", "src.models.cnn.predict", "c", "configs/cnn.yaml"])

        # Load model:
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name('cnn').experiment_id],
                                  order_by=["start_time desc"])
        latest_run_id = runs.iloc[0]["run_id"]
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/models")

        # Accuracy:
        f1 = float(f1_score(self.y_test, np.argmax(loaded_model.predict(self.x_test), axis=1), average='macro'))
        self.assertGreater(f1, 0.95, "F1 score should be greater than 95%")


if __name__ == '__main__':
    unittest.main()
