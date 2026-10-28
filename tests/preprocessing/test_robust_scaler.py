import numpy as np
import pytest

from dtaianomaly.preprocessing import RobustScaler


class TestRobustScaler:

    def test_invalid_quantile_ranges(self):
        with pytest.raises(ValueError):
            RobustScaler(75.0, 49.0)

    def test_default_quantile_range(self):
        X = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        robust_scaler = RobustScaler()

        robust_scaler.fit(X)
        assert robust_scaler.center_ == [15.0]
        assert robust_scaler.scale_ == [5.0]

        X_, _ = robust_scaler.transform(X)
        assert np.array_equal(
            X_, [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )

    def test_other_quantile_range(self):
        X = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        robust_scaler = RobustScaler(10.0, 90.0)

        robust_scaler.fit(X)
        assert robust_scaler.center_ == [15.0]
        assert robust_scaler.scale_ == [8.0]

        X_, _ = robust_scaler.transform(X)
        assert np.array_equal(
            X_,
            [-0.625, -0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625],
        )

    def test_single_value(self):
        X = np.ones(1000) * 123.4
        assert np.array_equal(RobustScaler().fit_transform(X)[0], X)

    def test_multivariate_with_single_value_attribute(self, multivariate_time_series):
        robust_scaler = RobustScaler()
        multivariate_time_series[:, 0] = 987.6
        X_, _ = robust_scaler.fit_transform(multivariate_time_series)
        assert np.array_equal(multivariate_time_series[:, 0], X_[:, 0])

    def test_nan_univariate(self):
        robust_scaler = RobustScaler().fit(np.r_[np.nan, np.arange(10.0)])
        assert np.array_equal(robust_scaler.scale_, [4.5])

    def test_nan_multivariate(self):
        X = np.c_[np.r_[np.nan, np.arange(10.0)], np.arange(11.0)]
        robust_scaler = RobustScaler().fit(X)
        assert np.array_equal(robust_scaler.scale_, [4.5, 5.0])

    def test_zero_iqr_univariate(self):
        X = np.r_[np.ones(98), 5.0, 6.0]
        X_, _ = RobustScaler().fit_transform(X)
        assert np.array_equal(X_, X)

    def test_zero_iqr_multivariate(self):
        X = np.c_[np.r_[np.ones(98), 5.0, 6.0], np.arange(100.0)]
        X_, _ = RobustScaler().fit_transform(X)
        assert np.array_equal(X_[:, 0], X[:, 0])
        assert np.allclose(X_[:, 1], (X[:, 1] - 49.5) / 49.5)

    @pytest.mark.parametrize(
        "fit_shape,transform_shape,n_attributes",
        [((100, 3), (50, 2), 2), ((100, 3), (50,), 1), ((100,), (50, 2), 2)],
    )
    def test_attribute_mismatch(self, fit_shape, transform_shape, n_attributes):
        rng = np.random.default_rng(42)
        robust_scaler = RobustScaler().fit(rng.standard_normal(fit_shape))
        with pytest.raises(AttributeError, match=f"with {n_attributes} attributes"):
            robust_scaler.transform(rng.standard_normal(transform_shape))
