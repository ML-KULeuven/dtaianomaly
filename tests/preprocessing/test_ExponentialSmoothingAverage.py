import pytest
import numpy as np

from dtaianomaly.preprocessing import ExponentialMovingAverage


class TestExponentialSmoothingAverage:

    def test_simple_univariate(self):
        preprocessor = ExponentialMovingAverage(0.5)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([1.0, 3.0, 3.0, 5.0, 6.5, 6.25, 5.125, 8.0625]))
        assert np.array_equal(y_, y)

    def test_simple_multivariate(self):
        preprocessor = ExponentialMovingAverage(0.5)

        x = np.array([[1, 10], [5, 50], [3, 30], [7, 70], [8, 80], [6, 60], [4, 40], [11, 110]])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([[1.0, 10.0], [3.0, 30.0], [3.0, 30.0], [5.0, 50.0], [6.5, 65.0], [6.25, 62.5], [5.125, 51.25], [8.0625, 80.625]]))
        assert np.array_equal(y_, y)
