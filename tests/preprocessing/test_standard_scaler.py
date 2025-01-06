import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from dtaianomaly.preprocessing import StandardScaler


class TestStandardScaler:

    def test(self):
        x = np.array([4, 9, 2, 5, 4, 7, 4, 5])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = StandardScaler()
        x_, y_ = preprocessor.fit_transform(x, y)
        assert x.shape == x_.shape
        assert y.shape == y_.shape
        assert np.array_equal(x_, np.array([-0.5, 2.0, -1.5, 0.0, -0.5, 1.0, -0.5, 0.0]))
        assert np.array_equal(y_, y)

    def test_distribution(self, univariate_time_series):
        x_, _ = StandardScaler().fit_transform(univariate_time_series)
        assert x_.mean() == pytest.approx(0.0)
        assert x_.std() == pytest.approx(1.0)

    def test_single_value(self):
        x = np.ones(1000) * 123.4
        assert np.array_equal(StandardScaler().fit_transform(x)[0], x)

    def test_multivariate(self, multivariate_time_series):
        preprocessor = StandardScaler()
        x_, _ = preprocessor.fit_transform(multivariate_time_series)
        for i in range(multivariate_time_series.shape[1]):
            assert x_[:, i].mean() == pytest.approx(0.0)
            assert x_[:, i].std() == pytest.approx(1.0)

    def test_multivariate_with_single_value_attribute(self, multivariate_time_series):
        preprocessor = StandardScaler()
        multivariate_time_series[:, 0] = 987.6
        x_, _ = preprocessor.fit_transform(multivariate_time_series)
        assert np.array_equal(multivariate_time_series[:, 0], x_[:, 0])
        for i in range(1, multivariate_time_series.shape[1]):
            assert x_[:, i].mean() == pytest.approx(0.0)
            assert x_[:, i].std() == pytest.approx(1.0)

    def test_different_dimension(self, univariate_time_series, multivariate_time_series):
        preprocessor = StandardScaler()
        preprocessor.fit(univariate_time_series)
        with pytest.raises(AttributeError):
            preprocessor.transform(multivariate_time_series)

    def test_not_fitted(self, univariate_time_series):
        preprocessor = StandardScaler()
        with pytest.raises(NotFittedError):
            preprocessor.transform(univariate_time_series)

    def test_str(self):
        assert str(StandardScaler()) == 'StandardScaler()'
