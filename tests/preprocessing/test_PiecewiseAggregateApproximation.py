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
        assert 0

    def test_simple_univariate(self):
        assert 0

    def test_simple_multivariate(self):
        assert 0

    def test_unequal_frames(self):
        assert 0

    def test_simple_no_y_given(self):
        assert 0

    def test_str(self):
        assert str(PiecewiseAggregateApproximation(32)) == 'PiecewiseAggregateApproximation(n=32)'
