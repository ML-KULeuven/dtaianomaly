
from dtaianomaly.anomaly_detection import LocalOutlierFactor


class TestLocalOutlierFactor:

    def test_str(self):
        assert str(LocalOutlierFactor(5)) == "LocalOutlierFactor(window_size=5)"
        assert str(LocalOutlierFactor(15, 3)) == "LocalOutlierFactor(window_size=15,stride=3)"
        assert str(LocalOutlierFactor(25, n_neighbors=42)) == "LocalOutlierFactor(window_size=25,n_neighbors=42)"
