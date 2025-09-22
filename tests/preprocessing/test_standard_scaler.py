import numpy as np
import pytest

from dtaianomaly.preprocessing import StandardScaler


class TestStandardScaler:

    def test_distribution(self, univariate_time_series):
        x_, _ = StandardScaler().fit_transform(univariate_time_series)
        assert x_.mean() == pytest.approx(0.0)
        assert x_.std() == pytest.approx(1.0)

    def test_single_value(self):
        x = np.ones(1000) * 123.4
        assert np.array_equal(StandardScaler().fit_transform(x)[0], x)

    def test_multivariate_with_single_value_attribute(self, multivariate_time_series):
        preprocessor = StandardScaler()
        multivariate_time_series[:, 0] = 987.6
        x_, _ = preprocessor.fit_transform(multivariate_time_series)
        assert np.array_equal(multivariate_time_series[:, 0], x_[:, 0])
        for i in range(1, multivariate_time_series.shape[1]):
            assert x_[:, i].mean() == pytest.approx(0.0)
            assert x_[:, i].std() == pytest.approx(1.0)
