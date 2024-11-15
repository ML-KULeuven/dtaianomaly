
import pytest
from dtaianomaly import anomaly_detection


@pytest.mark.parametrize('detector_class,kwargs', [
    (anomaly_detection.HistogramBasedOutlierScore, {'n_bins': 'auto', 'alpha': 0.5}),
    (anomaly_detection.IsolationForest, {'n_estimators': 42, 'max_samples': 'auto'}),
    (anomaly_detection.KNearestNeighbors, {'n_neighbors': 42, 'metric': 'euclidean'}),
    (anomaly_detection.LocalOutlierFactor, {'n_neighbors': 3}),
])
class TestPyodAnomalyDetectorAdditionalArgs:

    def test(self, detector_class, kwargs):
        detector = detector_class(window_size='fft', **kwargs)
        for key, value in kwargs.items():
            assert detector.kwargs[key] == value


@pytest.mark.parametrize('detector_class', [
    anomaly_detection.HistogramBasedOutlierScore,
    anomaly_detection.IsolationForest,
    anomaly_detection.KNearestNeighbors,
    anomaly_detection.LocalOutlierFactor,
])
class TestPyodAnomalyDetector:

    def test_initialize_too_small_window_size(self, detector_class):
        with pytest.raises(ValueError):
            detector_class(window_size=0)
        detector_class(1)  # Doesn't raise an error with float
        detector_class(15)  # Doesn't raise an error with int

    def test_initialize_float_window_size(self, detector_class):
        with pytest.raises(ValueError):
            detector_class(window_size=5.5)

    def test_initialize_valid_string_window_size(self, detector_class):
        detector_class(window_size='fft')

    def test_initialize_string_window_size(self, detector_class):
        with pytest.raises(ValueError):
            detector_class(window_size='15')

    def test_initialize_too_small_stride(self, detector_class):
        with pytest.raises(ValueError):
            detector_class(window_size=15, stride=0)
        detector_class(window_size=15, stride=1)  # Doesn't raise an error with float
        detector_class(window_size=15, stride=5)  # Doesn't raise an error with int

    def test_initialize_float_stride(self, detector_class):
        with pytest.raises(TypeError):
            detector_class(window_size=10, stride=2.5)

    def test_initialize_string_stride(self, detector_class):
        with pytest.raises(TypeError):
            detector_class(window_size=10, stride='1')

    def test_default_stride(self, detector_class):
        assert detector_class(window_size='fft').stride == 1

    def test_invalid_additional_arguments(self, detector_class):
        with pytest.raises(TypeError):
            detector_class(window_size='fft', some_invalid_arg=1)
