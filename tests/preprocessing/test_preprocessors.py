import inspect

import numpy as np
import pytest

from dtaianomaly import utils
from dtaianomaly.preprocessing import ChainedPreprocessor, Identity


def initialize(cls):
    if cls == ChainedPreprocessor:
        return ChainedPreprocessor(Identity(), Identity())
    kwargs = {
        "order": 2,
        "alpha": 0.7,
        "window_size": 15,
        "nb_samples": 500,
        "sampling_rate": 2,
        "n": 4,
    }
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(**filtered_kwargs)


@pytest.mark.parametrize("cls", utils.all_classes("preprocessor", return_names=False))
class TestPreprocessors:

    def test_is_valid_array_like_univariate_1d(self, cls, univariate_time_series):
        preprocessor = initialize(cls)
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X = univariate_time_series.squeeze()
        assert X.shape == (univariate_time_series.shape[0],)
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_univariate_2d(self, cls, univariate_time_series):
        preprocessor = initialize(cls)
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X = univariate_time_series.reshape(-1, 1)
        assert X.shape == (univariate_time_series.shape[0], 1)
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_univariate_list(self, cls, univariate_time_series):
        preprocessor = initialize(cls)
        ground_truth = np.zeros(univariate_time_series.shape[0])
        X = [v for v in univariate_time_series]
        assert len(X) == univariate_time_series.shape[0]
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_multivariate_2d(self, cls, multivariate_time_series):
        preprocessor = initialize(cls)
        ground_truth = np.zeros(multivariate_time_series.shape[0])
        X_, y_ = preprocessor.fit_transform(multivariate_time_series, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_is_valid_array_like_multivariate_list(self, cls, multivariate_time_series):
        preprocessor = initialize(cls)
        ground_truth = np.zeros(multivariate_time_series.shape[0])
        X = [list(v) for v in multivariate_time_series]
        assert len(X) == multivariate_time_series.shape[0]
        assert all(
            [len(observation) == multivariate_time_series.shape[1] for observation in X]
        )
        X_, y_ = preprocessor.fit_transform(X, ground_truth)
        assert utils.is_valid_array_like(X_)
        assert utils.is_valid_array_like(y_)

    def test_invalid_input_fit(self, cls):
        preprocessor = initialize(cls)
        with pytest.raises(ValueError):
            preprocessor.fit(
                np.array(["1", "2", "3", "4", "5"]), np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_input_transform(self, cls):
        preprocessor = initialize(cls)
        preprocessor.fit(np.array([1, 2, 3, 4, 5]), np.array([0, 0, 1, 0, 1]))
        with pytest.raises(ValueError):
            preprocessor.transform(
                np.array(["1", "2", "3", "4", "5"]), np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_input_fit_transform(self, cls):
        preprocessor = initialize(cls)
        with pytest.raises(ValueError):
            preprocessor.fit_transform(
                np.array(["1", "2", "3", "4", "5"]), np.array([0, 0, 1, 0, 1])
            )

    def test_fit_transform_different_time_series(self, cls, univariate_time_series):
        preprocessor = initialize(cls)
        split = int(univariate_time_series.shape[0] / 2)
        x_fit = univariate_time_series[:split]
        x_transform = univariate_time_series[split:]
        preprocessor.fit(x_fit).transform(x_transform)

    # def test_fit_transform_different_dimension(self, cls, univariate_time_series, multivariate_time_series):
    #     If does not require any fitting ...
    #     preprocessor = initialize(cls)
    #     preprocessor.fit(univariate_time_series)
    #     with pytest.raises(AttributeError):
    #         preprocessor.transform(multivariate_time_series)
