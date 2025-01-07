
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError

from dtaianomaly.preprocessing import RobustScaler


class TestRobustScaler:

    def test_valid_quantile_range(self):
        RobustScaler()
        RobustScaler(quantile_range=(10, 90))
        RobustScaler(quantile_range=(0.1, 99.9))

    def test_non_tuple_quantile_range(self):
        with pytest.raises(TypeError):
            RobustScaler(quantile_range=50)

    def test_too_short_quantile_range(self):
        with pytest.raises(ValueError):
            RobustScaler(quantile_range=(50,))

    def test_too_long_quantile_range(self):
        with pytest.raises(ValueError):
            RobustScaler(quantile_range=(25, 50, 75))

    def test_bool_quantile_range(self):
        with pytest.raises(TypeError):
            RobustScaler(quantile_range=(True, 90))
        with pytest.raises(TypeError):
            RobustScaler(quantile_range=(10, False))

    def test_str_quantile_range(self):
        with pytest.raises(TypeError):
            RobustScaler(quantile_range=('10', 90))
        with pytest.raises(TypeError):
            RobustScaler(quantile_range=(10, '90'))

    def test_too_small_q_min(self):
        with pytest.raises(ValueError):
            RobustScaler(quantile_range=(-10, 90))

    def test_too_big_q_max(self):
        with pytest.raises(ValueError):
            RobustScaler(quantile_range=(10, 110))

    def test_q_min_bigger_than_q_max(self):
        with pytest.raises(ValueError):
            RobustScaler(quantile_range=(90, 10))
        with pytest.raises(ValueError):
            RobustScaler(quantile_range=(50, 50))

    def test_default_quantile_range(self):
        X = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        robust_scaler = RobustScaler()

        robust_scaler.fit(X)
        assert robust_scaler.center_ == [15.0]
        assert robust_scaler.scale_ == [5.0]

        X_, _ = robust_scaler.transform(X)
        assert np.array_equal(X_, [-1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_other_quantile_range(self):
        X = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        robust_scaler = RobustScaler(quantile_range=(10, 90))

        robust_scaler.fit(X)
        assert robust_scaler.center_ == [15.0]
        assert robust_scaler.scale_ == [8.0]

        X_, _ = robust_scaler.transform(X)
        assert np.array_equal(X_, [-0.625, -0.5, -0.375, -0.25, -0.125, 0., 0.125, 0.25, 0.375, 0.5, 0.625])

    def test_2d_univariate(self):
        X = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).reshape(-1, 1)
        assert X.shape == (11, 1)

        robust_scaler = RobustScaler(quantile_range=(10, 90))
        X_, _ = robust_scaler.fit_transform(X)
        assert np.array_equal(X_, np.array([-0.625, -0.5, -0.375, -0.25, -0.125, 0., 0.125, 0.25, 0.375, 0.5, 0.625]).reshape(-1, 1))

    def test_multivariate(self):
        X = np.array([[0, 0], [1, 110], [2, 120], [3, 130], [4, 140], [5, 150], [6, 160], [7, 170], [8, 180], [9, 190], [10, 200]])
        assert X.shape == (11, 2)

        robust_scaler = RobustScaler()
        X_, _ = robust_scaler.fit_transform(X)
        assert np.array_equal(robust_scaler.center_, np.array([5., 150.]))
        assert np.array_equal(robust_scaler.scale_, np.array([5., 50.]))
        assert np.array_equal(X_, [[-1., -3.], [-0.8, -0.8], [-0.6, -0.6], [-0.4, -0.4], [-0.2, -0.2], [0., 0.], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1., 1.]])

    def test_single_value(self):
        X = np.ones(1000) * 123.4
        assert np.array_equal(RobustScaler().fit_transform(X)[0], X)

    def test_multivariate_with_single_value_attribute(self, multivariate_time_series):
        robust_scaler = RobustScaler()
        multivariate_time_series[:, 0] = 987.6
        X_, _ = robust_scaler.fit_transform(multivariate_time_series)
        assert np.array_equal(multivariate_time_series[:, 0], X_[:, 0])

    def test_transform_not_fitted(self, univariate_time_series):
        robust_scaler = RobustScaler()
        with pytest.raises(NotFittedError):
            robust_scaler.transform(univariate_time_series)

    def test_transform_different_dimension(self, univariate_time_series, multivariate_time_series):
        robust_scaler = RobustScaler()
        robust_scaler.fit(univariate_time_series)
        with pytest.raises(AttributeError):
            robust_scaler.transform(multivariate_time_series)

    def test_str(self):
        assert str(RobustScaler()) == 'RobustScaler()'
        assert str(RobustScaler(quantile_range=(10, 90))) == 'RobustScaler(quantile_range=(10, 90))'
