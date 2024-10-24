
import pytest
import copy
import numpy as np
from dtaianomaly import preprocessing, utils


@pytest.fixture(params=[
    preprocessing.Identity(),
    preprocessing.ExponentialMovingAverage(alpha=0.8),
    preprocessing.MinMaxScaler(),
    preprocessing.MovingAverage(window_size=15),
    preprocessing.NbSamplesUnderSampler(nb_samples=150),
    preprocessing.SamplingRateUnderSampler(sampling_rate=5),
    preprocessing.ZNormalizer(),
    preprocessing.Differencing(1)
])
def preprocessor(request):
    return copy.deepcopy(request.param)


class TestPreprocessors:

    def test_is_valid_array_like_univariate(self, preprocessor, univariate_time_series):
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X_, y_ = preprocessor.fit_transform(univariate_time_series, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_multivariate(self, preprocessor, multivariate_time_series):
        ground_truth = np.zeros(multivariate_time_series.shape[0])
        X_, y_ = preprocessor.fit_transform(multivariate_time_series, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)
