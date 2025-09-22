
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError

from dtaianomaly.preprocessing import MinMaxScaler


class TestMinMaxScaler:

    def test(self):
        x = np.array([1, 5, 3, 6, 9, 10, 3, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = MinMaxScaler()
        x_, y_ = preprocessor.fit_transform(x, y)
        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([0, 0.4, 0.2, 0.5, 0.8, 0.9, 0.2, 1.0]))
        assert np.array_equal(y_, y)

    def test_range(self, univariate_time_series):
        x_, _ = MinMaxScaler().fit_transform(univariate_time_series)
        assert x_.min() == 0
        assert x_.max() == 1

    def test_single_value(self):
        x = np.ones(1000) * 123.4
        assert np.array_equal(MinMaxScaler().fit_transform(x)[0], x)

    def test_multivariate_with_single_value_attribute(self, multivariate_time_series):
        preprocessor = MinMaxScaler()
        multivariate_time_series[:, 0] = 987.6
        x_, _ = preprocessor.fit_transform(multivariate_time_series)
        assert np.array_equal(multivariate_time_series[:, 0], x_[:, 0])
        for i in range(1, multivariate_time_series.shape[1]):
            assert x_[:, i].min() == 0
            assert x_[:, i].max() == 1
