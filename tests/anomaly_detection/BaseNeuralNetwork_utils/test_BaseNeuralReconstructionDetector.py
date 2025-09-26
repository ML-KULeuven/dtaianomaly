import numpy as np
import pytest

from dtaianomaly.anomaly_detection import AutoEncoder, ReconstructionDataset

_VALID_ERROR_METRICS = ["mean-absolute-error", "mean-squared-error"]


class TestInitialization:

    @pytest.mark.parametrize("error_metric", _VALID_ERROR_METRICS)
    def test_error_metric_valid(self, error_metric):
        detector = AutoEncoder(window_size=16, error_metric=error_metric)
        assert detector.error_metric == error_metric

    @pytest.mark.parametrize("error_metric", [32, 16.0, True])
    def test_error_metric_invalid_type(self, error_metric):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, error_metric=error_metric)

    @pytest.mark.parametrize("error_metric", ["invalid"])
    def test_error_metric_invalid_value(self, error_metric):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, error_metric=error_metric)


@pytest.mark.slow
class TestErrorMetric:

    @pytest.mark.parametrize("error_metric", _VALID_ERROR_METRICS)
    def test(self, error_metric, univariate_time_series):
        auto_encoder = AutoEncoder(
            window_size=12, error_metric=error_metric, n_epochs=1
        )
        auto_encoder.fit(univariate_time_series)
        auto_encoder.decision_function(univariate_time_series)

    @pytest.mark.parametrize("error_metric", _VALID_ERROR_METRICS)
    def test_multivariate(self, error_metric, multivariate_time_series):
        detector = AutoEncoder(window_size=12, error_metric=error_metric, n_epochs=1)
        detector.fit(multivariate_time_series)
        detector.decision_function(multivariate_time_series)

    def test_update_after_initialization(self, univariate_time_series):
        auto_encoder = AutoEncoder(
            window_size=12, error_metric="mean-absolute-error", n_epochs=1
        )
        auto_encoder.fit(univariate_time_series)

        auto_encoder.error_metric = "invalid"
        with pytest.raises(ValueError):
            auto_encoder.decision_function(univariate_time_series)


class TestBuildDataset:

    @pytest.mark.parametrize("seed", [0])
    @pytest.mark.parametrize("window_size", [1, 8])
    def test(self, seed, window_size):
        rng = np.random.default_rng(seed)
        t = rng.integers(100, 1000)
        n = rng.integers(1, 10)
        X = rng.uniform(size=(t, n))

        detector = AutoEncoder(window_size=window_size, standard_scaling=False)
        detector.window_size_ = detector.window_size
        dataset = detector._build_dataset(X)
        assert isinstance(dataset, ReconstructionDataset)
        assert len(dataset) == t - window_size + 1
        for i in range(t - window_size + 1):
            assert np.allclose(dataset[i][0].numpy(), X[i : i + detector.window_size_])
