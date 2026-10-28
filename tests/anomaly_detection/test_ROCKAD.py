import numpy as np

from dtaianomaly.anomaly_detection import ROCKAD, Supervision


class TestROCKAD:

    def test_supervision(self):
        assert ROCKAD(128).supervision == Supervision.UNSUPERVISED

    def test_seed(self, univariate_time_series):
        detector1 = ROCKAD(128, n_kernels=10, seed=0)
        y_pred1 = detector1.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        detector2 = ROCKAD(128, n_kernels=10, seed=0)
        y_pred2 = detector2.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        assert np.array_equal(y_pred1, y_pred2)

    def test_nearest_neighbors_fitted_on_bootstrap_samples(self):
        X = np.sin(np.arange(200) / 5)
        detector = ROCKAD(16, n_kernels=10, n_estimators=3, seed=0).fit(X)
        features = detector.power_transformer_.transform(
            detector.rocket_.transform(detector._sliding_window(X))
        )
        for nearest_neighbors in detector.nearest_neighbors_:
            assert nearest_neighbors._fit_X.shape == features.shape
            assert not np.array_equal(nearest_neighbors._fit_X, features)

    def test_nearest_neighbors_use_different_samples(self):
        X = np.sin(np.arange(200) / 5)
        detector = ROCKAD(16, n_kernels=10, n_estimators=2, seed=0).fit(X)
        assert not np.array_equal(
            detector.nearest_neighbors_[0]._fit_X,
            detector.nearest_neighbors_[1]._fit_X,
        )
