import numpy as np
import pytest

from dtaianomaly.anomaly_detection import ForecastDataset, MultilayerPerceptron
from dtaianomaly.anomaly_detection._BaseNeuralForecastingDetector import ERROR_METRICS
from dtaianomaly.windowing import reverse_sliding_window


class TestInitialize:

    def test_error_metric_valid(self):
        detector = MultilayerPerceptron(window_size=16, error_metric=ERROR_METRICS[-1])
        assert detector.error_metric == ERROR_METRICS[-1]

    def test_error_metric_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, error_metric=32)

    def test_error_metric_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, error_metric="invalid")

    def test_forecast_length_valid(self):
        detector = MultilayerPerceptron(window_size=16, forecast_length=8)
        assert detector.forecast_length == 8

    def test_forecast_length_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, forecast_length=16.0)

    def test_forecast_length_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, forecast_length=0)


@pytest.mark.slow
class TestErrorMetric:

    @pytest.mark.parametrize("error_metric", ERROR_METRICS)
    @pytest.mark.parametrize("forecast_length", [1, 8])
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
    @pytest.mark.parametrize("forecast_length", [1, 8])
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


STRIDE_FORECAST_LENGTH = [(1, 1), (2, 3), (3, 2), (4, 6), (5, 5), (10, 1), (10, 7)]


def _detector_with_mocked_evaluate(monkeypatch, stride, forecast_length, scores):
    detector = MultilayerPerceptron(
        window_size=10, stride=stride, forecast_length=forecast_length
    )
    detector.window_size_ = 10
    monkeypatch.setattr(detector, "_evaluate", lambda data_loader: scores(data_loader))
    return detector


class TestDecisionFunction:

    @pytest.mark.slow
    @pytest.mark.parametrize("stride,forecast_length", STRIDE_FORECAST_LENGTH)
    def test_shape(self, stride, forecast_length):
        X = np.random.default_rng(0).standard_normal(203)
        detector = MultilayerPerceptron(
            window_size=10,
            stride=stride,
            forecast_length=forecast_length,
            n_epochs=1,
            batch_size=256,
        )
        scores = detector.fit(X).decision_function(X)
        assert scores.shape == X.shape
        assert not np.isnan(scores).any()

    @pytest.mark.parametrize("stride,forecast_length", STRIDE_FORECAST_LENGTH)
    def test_alignment(self, monkeypatch, stride, forecast_length):
        X = np.random.default_rng(0).standard_normal(103)
        detector = _detector_with_mocked_evaluate(
            monkeypatch,
            stride,
            forecast_length,
            lambda data_loader: np.arange(len(data_loader.dataset), dtype=float),
        )
        n_windows = len(detector._build_dataset(X))
        expected = reverse_sliding_window(
            np.arange(n_windows, dtype=float), 10, stride, X.shape[0] - forecast_length
        )
        expected = np.concatenate((np.full(forecast_length, expected[0]), expected))
        assert np.array_equal(detector._decision_function(X), expected)

    @pytest.mark.parametrize("stride,forecast_length", STRIDE_FORECAST_LENGTH)
    def test_single_high_window(self, monkeypatch, stride, forecast_length):
        X = np.random.default_rng(0).standard_normal(103)

        def scores(data_loader):
            per_window = np.zeros(len(data_loader.dataset))
            per_window[3] = 1.0
            return per_window

        detector = _detector_with_mocked_evaluate(
            monkeypatch, stride, forecast_length, scores
        )
        start = 3 * stride + forecast_length
        assert np.array_equal(
            np.flatnonzero(detector._decision_function(X)),
            np.arange(start, start + 10),
        )
