
import pytest
import numpy as np
from dtaianomaly.preprocessing import Differencing


class TestExponentialSmoothingAverage:

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            Differencing(-1)
        Differencing(0)
        Differencing(1)
        Differencing(2)

    def test_str_order(self):
        with pytest.raises(TypeError):
            Differencing('1')

    def test_bool_order(self):
        with pytest.raises(TypeError):
            Differencing(True)

    def test_float_order(self):
        with pytest.raises(TypeError):
            Differencing(1.0)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            Differencing(1, 0)
        Differencing(1, 1)
        Differencing(1, 2)
        Differencing(1, 20)

    def test_str_window_size(self):
        with pytest.raises(TypeError):
            Differencing(1, '1')

    def test_bool_window_size(self):
        with pytest.raises(TypeError):
            Differencing(1, True)

    def test_float_window_size(self):
        with pytest.raises(TypeError):
            Differencing(1, 1.0)

    def test_simple_univariate(self):
        preprocessor = Differencing(1)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([0, 4, -2, 4, 1, -2, -2, 7]))
        assert np.array_equal(y_, y)

    def test_simple_univariate_seasonal(self):
        preprocessor = Differencing(1, 2)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([0, 0, 2, 2, 5, -1, -4, 5]))
        assert np.array_equal(y_, y)

    def test_simple_multivariate(self):
        preprocessor = Differencing(1)

        x = np.array([[1, 10], [5, 50], [3, 30], [7, 70], [8, 80], [6, 60], [4, 40], [11, 110]])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([[0, 0], [4, 40], [-2, -20], [4, 40], [1, 10], [-2, -20], [-2, -20], [7, 70]]))
        assert np.array_equal(y_, y)

    def test_simple_multivariate_seasonal(self):
        preprocessor = Differencing(1, 2)

        x = np.array([[1, 10], [5, 50], [3, 30], [7, 70], [8, 80], [6, 60], [4, 40], [11, 110]])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([[0, 0], [0, 0], [2, 20], [2, 20], [5, 50], [-1, -10], [-4, -40], [5, 50]]))
        assert np.array_equal(y_, y)

    def test_order_zero(self, univariate_time_series):
        preprocessor = Differencing(0)
        assert np.array_equal(univariate_time_series, preprocessor.fit_transform(univariate_time_series)[0])

    def test_simple_order_two(self):
        preprocessor = Differencing(2)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([0, 4, -6, 6, -3, -3, 0, 9]))
        assert np.array_equal(y_, y)

    @pytest.mark.parametrize("window_size", [1, 10])
    def test_order_two(self, univariate_time_series, window_size):
        preprocessing_1 = Differencing(1, window_size=window_size)
        preprocessing_2 = Differencing(2, window_size=window_size)
        x_0 = univariate_time_series
        x_1, _ = preprocessing_1.fit_transform(x_0)
        x_1_1, _ = preprocessing_1.transform(x_1)
        x_2, _ = preprocessing_2.transform(x_0)
        assert np.array_equal(x_1_1, x_2)

    def test_univariate(self, univariate_time_series):
        x_, _ = Differencing(1).fit_transform(univariate_time_series)
        assert x_.shape == univariate_time_series.shape

    def test_multivariate(self, multivariate_time_series):
        x_, _ = Differencing(1).fit_transform(multivariate_time_series)
        assert x_.shape == multivariate_time_series.shape

    def test_str(self):
        assert str(Differencing(1)) == 'Differencing(order=1)'
        assert str(Differencing(1, 2)) == 'Differencing(order=1,window_size=2)'
