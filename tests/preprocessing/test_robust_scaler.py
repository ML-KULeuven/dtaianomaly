import numpy as np

from dtaianomaly.preprocessing import RobustScaler


class TestRobustScaler:

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
