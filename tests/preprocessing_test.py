import unittest

import numpy as np

from src.models.cnn.preprocessing import onehot_encoding, scale, reshape


class TestFunction(unittest.TestCase):

    def test_onehot_encoding(self):
        # Test 1: Check one-hot encoding for a single integer
        feature1 = 3
        result1 = onehot_encoding(feature1)
        expected_result1 = np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
        np.testing.assert_array_equal(result1, expected_result1)

        # Test 2: Check one-hot encoding for an array of integers
        feature2 = np.array([1, 5, 9])
        result2 = onehot_encoding(feature2)
        expected_result2 = np.array([
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        ])
        np.testing.assert_array_equal(result2, expected_result2)

        # Test 3: Check one-hot encoding with a different number of categories
        feature3 = np.array([2, 7])
        result3 = onehot_encoding(feature3, num_cat=8)
        expected_result3 = np.array([
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.]
        ])
        np.testing.assert_array_equal(result3, expected_result3)

    def test_scale(self):
        # Test 1: Test default factor:
        input_data = np.array([1, 2, 3], dtype=np.uint8)
        expected_output = np.array([1 / 255, 2 / 255, 3 / 255], dtype=np.float32)

        scaled_data = scale(input_data)
        self.assertTrue(np.array_equal(scaled_data, expected_output))

        # Test 2: Test with different factor:
        custom_factor = 10
        expected_output = np.array([1 / 10, 2 / 10, 3 / 10], dtype=np.float32)

        scaled_data = scale(input_data, factor=custom_factor)

        self.assertTrue(np.array_equal(scaled_data, expected_output))

    def test_reshape(self):
        # Test 1: (x, 784)
        input_data_1 = np.random.rand(15, 784)
        reshaped_data_1 = reshape(input_data_1)
        self.assertEqual(reshaped_data_1.shape, (15, 28, 28, 1))

        # Test 2: (x, 28, 28)
        input_data_2 = np.random.rand(10, 28, 28)
        reshaped_data_2 = reshape(input_data_2)
        self.assertEqual(reshaped_data_2.shape, (10, 28, 28, 1))

        # Test 3: (x, 28, 28, 1)
        input_data_3 = np.random.rand(12, 28, 28, 1)
        reshaped_data_3 = reshape(input_data_3)
        self.assertEqual(reshaped_data_3.shape, (12, 28, 28, 1))

        # Test 4: (28, 28)
        input_data_4 = np.random.rand(28, 28)
        reshaped_data_4 = reshape(input_data_4)
        self.assertEqual(reshaped_data_4.shape, (1, 28, 28, 1))

        # Test 5: (28, 28, 1)
        input_data_5 = np.random.rand(28, 28, 1)
        reshaped_data_5 = reshape(input_data_5)
        self.assertEqual(reshaped_data_5.shape, (1, 28, 28, 1))


if __name__ == '__main__':
    unittest.main()
