
import pytest
from dtaianomaly.anomaly_detection import LocalOutlierFactor


class TestLocalOutlierFactor:

    def test_initialize(self):
        detector = LocalOutlierFactor(15, n_neighbors=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.kwargs['n_neighbors'] == 42

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            LocalOutlierFactor(window_size=0)
        LocalOutlierFactor(1)  # Doesn't raise an error with float
        LocalOutlierFactor(15)  # Doesn't raise an error with int

    def test_initialize_float_window_size(self):
        with pytest.raises(ValueError):
            LocalOutlierFactor(window_size=5.5)

    def test_initialize_valid_string_window_size(self):
        LocalOutlierFactor(window_size='fft')

    def test_initialize_string_window_size(self):
        with pytest.raises(ValueError):
            LocalOutlierFactor(window_size='15')

    def test_initialize_too_small_stride(self):
        with pytest.raises(ValueError):
            LocalOutlierFactor(window_size=15, stride=0)
        LocalOutlierFactor(window_size=15, stride=1)  # Doesn't raise an error with float
        LocalOutlierFactor(window_size=15, stride=5)  # Doesn't raise an error with int

    def test_initialize_float_stride(self):
        with pytest.raises(TypeError):
            LocalOutlierFactor(window_size=10, stride=2.5)

    def test_initialize_string_stride(self):
        with pytest.raises(TypeError):
            LocalOutlierFactor(window_size=10, stride='1')

    def test_str(self):
        assert str(LocalOutlierFactor(5)) == "LocalOutlierFactor(window_size=5)"
        assert str(LocalOutlierFactor(15, 3)) == "LocalOutlierFactor(window_size=15,stride=3)"
        assert str(LocalOutlierFactor(25, n_neighbors=42)) == "LocalOutlierFactor(window_size=25,n_neighbors=42)"
