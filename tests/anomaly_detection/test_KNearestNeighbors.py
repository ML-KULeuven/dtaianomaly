
from dtaianomaly.anomaly_detection import KNearestNeighbors, Supervision


class TestKNearestNeighbors:

    def test_supervision(self):
        detector = KNearestNeighbors(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(KNearestNeighbors(5)) == "KNearestNeighbors(window_size=5)"
        assert str(KNearestNeighbors('fft')) == "KNearestNeighbors(window_size='fft')"
        assert str(KNearestNeighbors(15, 3)) == "KNearestNeighbors(window_size=15,stride=3)"
        assert str(KNearestNeighbors(25, n_neighbors=42)) == "KNearestNeighbors(window_size=25,n_neighbors=42)"
        assert str(KNearestNeighbors(25, metric='euclidean')) == "KNearestNeighbors(window_size=25,metric='euclidean')"
