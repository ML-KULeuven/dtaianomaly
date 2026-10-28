import pytest

from dtaianomaly import anomaly_detection
from dtaianomaly.utils import all_classes

_PYOD_DETECTOR_CLASSES = all_classes(
    anomaly_detection.BasePyODAnomalyDetector, return_names=False
)


@pytest.mark.parametrize("detector_class", _PYOD_DETECTOR_CLASSES)
class TestPyodAnomalyDetectorPerClass:

    def test_initialize_valid_string_window_size(self, detector_class):
        detector_class(window_size="fft")

    def test_invalid_additional_arguments(self, detector_class):
        with pytest.raises(TypeError):
            detector_class(window_size="fft", some_invalid_arg=1)


class TestPyodAnomalyDetector:
    # Window size and stride validation lives in BasePyODAnomalyDetector, so one
    # representative subclass suffices
    detector_class = anomaly_detection.IsolationForest

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            self.detector_class(window_size=0)
        self.detector_class(1)
        self.detector_class(15)

    def test_initialize_float_window_size(self):
        with pytest.raises(TypeError):
            self.detector_class(window_size=5.5)

    def test_initialize_string_window_size(self):
        with pytest.raises(ValueError):
            self.detector_class(window_size="15")

    def test_initialize_too_small_stride(self):
        with pytest.raises(ValueError):
            self.detector_class(window_size=15, stride=0)
        self.detector_class(window_size=15, stride=1)
        self.detector_class(window_size=15, stride=5)

    def test_initialize_float_stride(self):
        with pytest.raises(TypeError):
            self.detector_class(window_size=10, stride=2.5)

    def test_initialize_string_stride(self):
        with pytest.raises(TypeError):
            self.detector_class(window_size=10, stride="1")

    def test_default_stride(self):
        assert self.detector_class(window_size="fft").stride == 1
