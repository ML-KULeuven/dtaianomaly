
import pytest
from dtaianomaly.anomaly_detection import KMeansAnomalyDetector, Supervision


class TestKMeansAnomalyDetector:

    def test_supervision(self):
        detector = KMeansAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_initialize_invalid_stride(self):
        with pytest.raises(ValueError):
            KMeansAnomalyDetector(window_size=15, stride=0)
        with pytest.raises(TypeError):
            KMeansAnomalyDetector(window_size=10, stride=2.5)
        with pytest.raises(TypeError):
            KMeansAnomalyDetector(window_size=10, stride='1')
        KMeansAnomalyDetector(window_size=15, stride=1)
        KMeansAnomalyDetector(window_size=15, stride=5)

    def test_str(self):
        assert str(KMeansAnomalyDetector(5)) == "KMeansAnomalyDetector(window_size=5)"
        assert str(KMeansAnomalyDetector('fft')) == "KMeansAnomalyDetector(window_size='fft')"
        assert str(KMeansAnomalyDetector(15, 3)) == "KMeansAnomalyDetector(window_size=15,stride=3)"
        assert str(KMeansAnomalyDetector(25, n_clusters=3)) == "KMeansAnomalyDetector(window_size=25,n_clusters=3)"

    def test_n_clusters(self):
        KMeansAnomalyDetector(15, n_clusters=2)
        KMeansAnomalyDetector(15, n_clusters=4)
        KMeansAnomalyDetector(15, n_clusters=8)
        with pytest.raises(TypeError):
            KMeansAnomalyDetector(15, n_clusters='6')
        with pytest.raises(TypeError):
            KMeansAnomalyDetector(15, n_clusters=6.0)
        with pytest.raises(TypeError):
            KMeansAnomalyDetector(15, n_clusters=True)
        with pytest.raises(ValueError):
            KMeansAnomalyDetector(15, n_clusters=0)
        with pytest.raises(ValueError):
            KMeansAnomalyDetector(15, n_clusters=-1)
        with pytest.raises(ValueError):
            KMeansAnomalyDetector(15, n_clusters=1)