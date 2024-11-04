import numpy as np
import pytest
from dtaianomaly.preprocessing import PiecewiseAggregateApproximation


class TestPiecewiseAggregateApproximation:

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            PiecewiseAggregateApproximation(0)
        PiecewiseAggregateApproximation(1)
        PiecewiseAggregateApproximation(2)

    def test_str_order(self):
        with pytest.raises(TypeError):
            PiecewiseAggregateApproximation('1')

    def test_bool_order(self):
        with pytest.raises(TypeError):
            PiecewiseAggregateApproximation(True)

    def test_float_order(self):
        with pytest.raises(TypeError):
            PiecewiseAggregateApproximation(1.0)

    def test_n_equals_1_univariate(self, univariate_time_series):
        preprocessor = PiecewiseAggregateApproximation(1)
        X_, _ = preprocessor.fit_transform(univariate_time_series)
        assert np.array_equal(X_, [np.mean(univariate_time_series, axis=0)])

    def test_n_equals_1_multivariate(self, multivariate_time_series):
        preprocessor = PiecewiseAggregateApproximation(1)
        X_, _ = preprocessor.fit_transform(multivariate_time_series)
        assert np.array_equal(X_, [np.mean(multivariate_time_series, axis=0)])

    def test_too_short_time_series(self):
        preprocessor = PiecewiseAggregateApproximation(500)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])  # Shorter than 'n' in PAA
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, x)
        assert np.array_equal(y_, y)

    def test_time_series_length_equals_n(self):
        preprocessor = PiecewiseAggregateApproximation(8)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, x)
        assert np.array_equal(y_, y)

    def test_simple_univariate(self):
        preprocessor = PiecewiseAggregateApproximation(4)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x_.shape == (4,)
        assert y_.shape == (4,)
        assert np.array_equal(x_, np.array([3, 5, 7, 7.5]))
        assert np.array_equal(y_, np.array([1, 0, 1, 1]))

    def test_simple_multivariate(self):
        preprocessor = PiecewiseAggregateApproximation(4)

        x = np.array([[1, 10], [5, 50], [3, 30], [7, 70], [8, 80], [6, 60], [4, 40], [11, 110]])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x_.shape == (4, 2)
        assert y_.shape == (4,)
        assert np.array_equal(x_, np.array([[3, 30], [5, 50], [7, 70], [7.5, 75]]))
        assert np.array_equal(y_, np.array([1, 0, 1, 1]))

    def test_unequal_frames(self):
        preprocessor = PiecewiseAggregateApproximation(4)

        x = np.array([1, 5, 3, 7, 8, 6, 4, 11, 9])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0])
        x_, y_ = preprocessor.fit_transform(x, y)

        assert x_.shape == (4,)
        assert y_.shape == (4,)
        assert np.array_equal(x_, np.array([3, 5, 7, 8]))
        assert np.array_equal(y_, np.array([1, 0, 1, 0]))

    def test_simple_no_y_given(self, univariate_time_series):
        n = int(univariate_time_series.shape[0] * 0.25)
        preprocessor = PiecewiseAggregateApproximation(n)

        x_, y_ = preprocessor.fit_transform(univariate_time_series)

        assert x_.shape[0] == n
        assert y_ is None

    def test_str(self):
        assert str(PiecewiseAggregateApproximation(32)) == 'PiecewiseAggregateApproximation(n=32)'
