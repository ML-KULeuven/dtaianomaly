import pytest

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

    def test_n_clusters(self):
        ClusterBasedLocalOutlierFactor(15, n_clusters=2)
        ClusterBasedLocalOutlierFactor(15, n_clusters=4)
        ClusterBasedLocalOutlierFactor(15, n_clusters=8)
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, n_clusters='6')
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, n_clusters=6.0)
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, n_clusters=True)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, n_clusters=0)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, n_clusters=-1)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, n_clusters=1)

    def test_alpha(self):
        ClusterBasedLocalOutlierFactor(15, alpha=0.5)
        ClusterBasedLocalOutlierFactor(15, alpha=1)
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, alpha='0.5')
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, alpha=True)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, alpha=-0.1)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, alpha=-1)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, alpha=1.1)

    def test_beta(self):
        ClusterBasedLocalOutlierFactor(15, beta=10)
        ClusterBasedLocalOutlierFactor(15, beta=4.3)
        ClusterBasedLocalOutlierFactor(15, beta=1.0)
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, beta='5.0')
        with pytest.raises(TypeError):
            ClusterBasedLocalOutlierFactor(15, beta=True)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, beta=0.9)
        with pytest.raises(ValueError):
            ClusterBasedLocalOutlierFactor(15, beta=-1)
