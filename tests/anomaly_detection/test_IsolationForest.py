
from dtaianomaly.anomaly_detection import IsolationForest


class TestIsolationForest:

    def test_str(self):
        assert str(IsolationForest(5)) == "IsolationForest(window_size=5)"
        assert str(IsolationForest('fft')) == "IsolationForest(window_size='fft')"
        assert str(IsolationForest(15, 3)) == "IsolationForest(window_size=15,stride=3)"
        assert str(IsolationForest(25, n_estimators=42)) == "IsolationForest(window_size=25,n_estimators=42)"
        assert str(IsolationForest(25, max_samples=50)) == "IsolationForest(window_size=25,max_samples=50)"
        assert str(IsolationForest(25, max_samples='auto')) == "IsolationForest(window_size=25,max_samples='auto')"
