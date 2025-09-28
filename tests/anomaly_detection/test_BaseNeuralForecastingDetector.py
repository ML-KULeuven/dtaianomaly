import numpy as np
import pytest

from dtaianomaly.anomaly_detection import ForecastDataset, MultilayerPerceptron
from dtaianomaly.anomaly_detection._BaseNeuralForecastingDetector import ERROR_METRICS


class TestInitialize:

    @pytest.mark.parametrize("error_metric", ERROR_METRICS)
    def test_error_metric_valid(self, error_metric):
        detector = MultilayerPerceptron(window_size=16, error_metric=error_metric)
        assert detector.error_metric == error_metric

    @pytest.mark.parametrize("error_metric", [32, 16.0, True])
    def test_error_metric_invalid_type(self, error_metric):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, error_metric=error_metric)

    @pytest.mark.parametrize("error_metric", ["invalid"])
    def test_error_metric_invalid_value(self, error_metric):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, error_metric=error_metric)

    @pytest.mark.parametrize("forecast_length", [32, 16, 8])
    def test_forecast_length_valid(self, forecast_length):
        detector = MultilayerPerceptron(window_size=16, forecast_length=forecast_length)
        assert detector.forecast_length == forecast_length

    @pytest.mark.parametrize("forecast_length", ["32", 16.0, True])
    def test_forecast_length_invalid_type(self, forecast_length):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, forecast_length=forecast_length)

    @pytest.mark.parametrize("forecast_length", [0, -1, -16])
    def test_forecast_length_invalid_value(self, forecast_length):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, forecast_length=forecast_length)


@pytest.mark.slow
class TestErrorMetric:

    @pytest.mark.parametrize("error_metric", ERROR_METRICS)
    @pytest.mark.parametrize("forecast_length", [1, 4, 8])
    def test(self, error_metric, forecast_length, univariate_time_series):
        detector = MultilayerPerceptron(
            window_size=12,
            forecast_length=forecast_length,
            error_metric=error_metric,
            n_epochs=1,
        )
        detector.fit(univariate_time_series)
        detector.decision_function(univariate_time_series)

    @pytest.mark.parametrize("error_metric", ERROR_METRICS)
    @pytest.mark.parametrize("forecast_length", [1, 4, 8])
    def test_multivariate(
        self, error_metric, forecast_length, multivariate_time_series
    ):
        detector = MultilayerPerceptron(
            window_size=12,
            forecast_length=forecast_length,
            error_metric=error_metric,
            n_epochs=1,
            data_loader_kwargs={"drop_last": True},
        )
        detector.fit(multivariate_time_series)
        detector.decision_function(multivariate_time_series)


class TestBuildDataset:

    @pytest.mark.parametrize("seed", [0])
    @pytest.mark.parametrize("window_size", [1, 8])
    @pytest.mark.parametrize("forecast_length", [1, 4])
    def test(self, seed, window_size, forecast_length):
        rng = np.random.default_rng(seed)
        t = rng.integers(100, 1000)
        n = rng.integers(1, 10)
        X = rng.uniform(size=(t, n))

        detector = MultilayerPerceptron(
            window_size=window_size,
            forecast_length=forecast_length,
            standard_scaling=False,
        )
        detector.window_size_ = detector.window_size
        dataset = detector._build_dataset(X)
        assert isinstance(dataset, ForecastDataset)
        assert len(dataset) == t - window_size - forecast_length + 1
        for i in range(t - window_size - forecast_length + 1):
            history, future = dataset[i]
            assert np.allclose(history.numpy(), X[i : i + detector.window_size_])
            assert np.allclose(
                future.numpy(),
                X[
                    i
                    + detector.window_size_ : i
                    + detector.window_size_
                    + detector.forecast_length
                ],
            )
