import pytest

from dtaianomaly import anomaly_detection
from dtaianomaly.utils import all_classes


@pytest.mark.parametrize(
    "detector_class",
    all_classes(anomaly_detection.BasePyODAnomalyDetector, return_names=False),
)
class TestPyodAnomalyDetector:

    def test_initialize_too_small_window_size(self, detector_class):
        with pytest.raises(ValueError):
            detector_class(window_size=0)
        detector_class(1)  # Doesn't raise an error with float
        detector_class(15)  # Doesn't raise an error with int

    def test_initialize_float_window_size(self, detector_class):
        with pytest.raises(TypeError):
            detector_class(window_size=5.5)

    def test_initialize_valid_string_window_size(self, detector_class):
        detector_class(window_size="fft")

    def test_initialize_string_window_size(self, detector_class):
        with pytest.raises(ValueError):
            detector_class(window_size="15")

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
            detector_class(window_size=10, stride="1")

    def test_default_stride(self, detector_class):
        assert detector_class(window_size="fft").stride == 1

    def test_invalid_additional_arguments(self, detector_class):
        with pytest.raises(TypeError):
            detector_class(window_size="fft", some_invalid_arg=1)
