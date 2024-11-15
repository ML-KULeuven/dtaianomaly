
import pytest
from dtaianomaly.anomaly_detection import HistogramBasedOutlierScore


class TestIsolationForest:

    def test_initialize(self):
        detector = HistogramBasedOutlierScore(15, n_bins=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.kwargs['n_bins'] == 42

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(window_size=0)
        HistogramBasedOutlierScore(1)  # Doesn't raise an error with float
        HistogramBasedOutlierScore(15)  # Doesn't raise an error with int

    def test_initialize_float_window_size(self):
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(window_size=5.5)

    def test_initialize_valid_string_window_size(self):
        HistogramBasedOutlierScore(window_size='fft')

    def test_initialize_string_window_size(self):
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(window_size='15')

    def test_initialize_too_small_stride(self):
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(window_size=15, stride=0)
        HistogramBasedOutlierScore(window_size=15, stride=1)  # Doesn't raise an error with float
        HistogramBasedOutlierScore(window_size=15, stride=5)  # Doesn't raise an error with int

    def test_initialize_float_stride(self):
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(window_size=10, stride=2.5)

    def test_initialize_string_stride(self):
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(window_size=10, stride='1')

    def test_str(self):
        assert str(HistogramBasedOutlierScore(5)) == "HistogramBasedOutlierScore(window_size=5)"
        assert str(HistogramBasedOutlierScore('fft')) == "HistogramBasedOutlierScore(window_size='fft')"
        assert str(HistogramBasedOutlierScore(15, 3)) == "HistogramBasedOutlierScore(window_size=15,stride=3)"
        assert str(HistogramBasedOutlierScore(25, n_bins=42)) == "HistogramBasedOutlierScore(window_size=25,n_bins=42)"
        assert str(HistogramBasedOutlierScore(25, alpha=0.5)) == "HistogramBasedOutlierScore(window_size=25,alpha=0.5)"
