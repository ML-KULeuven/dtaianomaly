
import pytest
from dtaianomaly.anomaly_detection import IsolationForest


class TestIsolationForest:

    def test_initialize(self):
        detector = IsolationForest(15, n_estimators=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.kwargs['n_estimators'] == 42

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            IsolationForest(window_size=0)
        IsolationForest(1)  # Doesn't raise an error with float
        IsolationForest(15)  # Doesn't raise an error with int

    def test_initialize_float_window_size(self):
        with pytest.raises(ValueError):
            IsolationForest(window_size=5.5)

    def test_initialize_valid_string_window_size(self):
        IsolationForest(window_size='fft')

    def test_initialize_string_window_size(self):
        with pytest.raises(ValueError):
            IsolationForest(window_size='15')

    def test_initialize_too_small_stride(self):
        with pytest.raises(ValueError):
            IsolationForest(window_size=15, stride=0)
        IsolationForest(window_size=15, stride=1)  # Doesn't raise an error with float
        IsolationForest(window_size=15, stride=5)  # Doesn't raise an error with int

    def test_initialize_float_stride(self):
        with pytest.raises(TypeError):
            IsolationForest(window_size=10, stride=2.5)

    def test_initialize_string_stride(self):
        with pytest.raises(TypeError):
            IsolationForest(window_size=10, stride='1')

    def test_str(self):
        assert str(IsolationForest(5)) == "IsolationForest(window_size=5)"
        assert str(IsolationForest('fft')) == "IsolationForest(window_size='fft')"
        assert str(IsolationForest(15, 3)) == "IsolationForest(window_size=15,stride=3)"
        assert str(IsolationForest(25, n_estimators=42)) == "IsolationForest(window_size=25,n_estimators=42)"
        assert str(IsolationForest(25, max_samples=50)) == "IsolationForest(window_size=25,max_samples=50)"
        assert str(IsolationForest(25, max_samples='auto')) == "IsolationForest(window_size=25,max_samples='auto')"
