
import pytest
from dtaianomaly.anomaly_detection import KNearestNeighbors


class TestKNearestNeighbors:

    def test_initialize(self):
        detector = KNearestNeighbors(15, n_neighbors=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.kwargs['n_neighbors'] == 42

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            KNearestNeighbors(window_size=0)
        KNearestNeighbors(1)  # Doesn't raise an error with float
        KNearestNeighbors(15)  # Doesn't raise an error with int

    def test_initialize_float_window_size(self):
        with pytest.raises(ValueError):
            KNearestNeighbors(window_size=5.5)

    def test_initialize_valid_string_window_size(self):
        KNearestNeighbors(window_size='fft')

    def test_initialize_string_window_size(self):
        with pytest.raises(ValueError):
            KNearestNeighbors(window_size='15')

    def test_initialize_too_small_stride(self):
        with pytest.raises(ValueError):
            KNearestNeighbors(window_size=15, stride=0)
        KNearestNeighbors(window_size=15, stride=1)  # Doesn't raise an error with float
        KNearestNeighbors(window_size=15, stride=5)  # Doesn't raise an error with int

    def test_initialize_float_stride(self):
        with pytest.raises(TypeError):
            KNearestNeighbors(window_size=10, stride=2.5)

    def test_initialize_string_stride(self):
        with pytest.raises(TypeError):
            KNearestNeighbors(window_size=10, stride='1')

    def test_str(self):
        assert str(KNearestNeighbors(5)) == "KNearestNeighbors(window_size=5)"
        assert str(KNearestNeighbors('fft')) == "KNearestNeighbors(window_size='fft')"
        assert str(KNearestNeighbors(15, 3)) == "KNearestNeighbors(window_size=15,stride=3)"
        assert str(KNearestNeighbors(25, n_neighbors=42)) == "KNearestNeighbors(window_size=25,n_neighbors=42)"
        assert str(KNearestNeighbors(25, metric='euclidean')) == "KNearestNeighbors(window_size=25,metric='euclidean')"
