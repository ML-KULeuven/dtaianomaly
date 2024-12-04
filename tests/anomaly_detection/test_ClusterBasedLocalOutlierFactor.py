
from dtaianomaly.anomaly_detection import ClusterBasedLocalOutlierFactor, Supervision


class TestClusterBasedLocalOutlierFactor:

    def test_supervision(self):
        detector = ClusterBasedLocalOutlierFactor(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(ClusterBasedLocalOutlierFactor(5)) == "ClusterBasedLocalOutlierFactor(window_size=5)"
        assert str(ClusterBasedLocalOutlierFactor('fft')) == "ClusterBasedLocalOutlierFactor(window_size='fft')"
        assert str(ClusterBasedLocalOutlierFactor(15, 3)) == "ClusterBasedLocalOutlierFactor(window_size=15,stride=3)"
        assert str(ClusterBasedLocalOutlierFactor(25, n_clusters=3)) == "ClusterBasedLocalOutlierFactor(window_size=25,n_clusters=3)"
        assert str(ClusterBasedLocalOutlierFactor(25, alpha=0.5)) == "ClusterBasedLocalOutlierFactor(window_size=25,alpha=0.5)"
