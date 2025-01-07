
import pytest
import copy
import numpy as np
from dtaianomaly import preprocessing, utils


@pytest.fixture(params=[
    preprocessing.Identity(),
    preprocessing.ChainedPreprocessor(preprocessing.Identity(), preprocessing.Identity()),
    preprocessing.ExponentialMovingAverage(alpha=0.8),
    preprocessing.MinMaxScaler(),
    preprocessing.MovingAverage(window_size=15),
    preprocessing.NbSamplesUnderSampler(nb_samples=150),
    preprocessing.SamplingRateUnderSampler(sampling_rate=5),
    preprocessing.StandardScaler(),
    preprocessing.RobustScaler(),
    preprocessing.Differencing(order=1),
    preprocessing.PiecewiseAggregateApproximation(n=32)
])
def preprocessor(request):
    return copy.deepcopy(request.param)


class TestPreprocessors:

    def test_is_valid_array_like_univariate_1d(self, preprocessor, univariate_time_series):
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X = univariate_time_series.squeeze()
        assert X.shape == (univariate_time_series.shape[0],)
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_univariate_2d(self, preprocessor, univariate_time_series):
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X = univariate_time_series.reshape(-1, 1)
        assert X.shape == (univariate_time_series.shape[0], 1)
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_univariate_list(self, preprocessor, univariate_time_series):
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X = [v for v in univariate_time_series]
        assert len(X) == univariate_time_series.shape[0]
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_multivariate_2d(self, preprocessor, multivariate_time_series):
        ground_truth = np.zeros(multivariate_time_series.shape[0])
        X_, y_ = preprocessor.fit_transform(multivariate_time_series, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_multivariate_list(self, preprocessor, multivariate_time_series):
        ground_truth = np.zeros(multivariate_time_series.shape[0])
        X = [list(v) for v in multivariate_time_series]
        assert len(X) == multivariate_time_series.shape[0]
        assert all([len(observation) == multivariate_time_series.shape[1] for observation in X])
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_invalid_input_fit(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.fit(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_input_transform(self, preprocessor):
        preprocessor.fit(
            np.array([1, 2, 3, 4, 5]),
            np.array([0, 0, 1, 0, 1])
        )
        with pytest.raises(ValueError):
            preprocessor.transform(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_input_fit_transform(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.fit_transform(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_fit_transform_different_time_series(self, preprocessor, univariate_time_series):
        split = int(univariate_time_series.shape[0] / 2)
        x_fit = univariate_time_series[:split]
        x_transform = univariate_time_series[split:]
        preprocessor.fit(x_fit).transform(x_transform)
