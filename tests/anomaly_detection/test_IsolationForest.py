import pytest

from dtaianomaly.anomaly_detection import IsolationForest, Supervision


class TestIsolationForest:

    def test_supervision(self):
        detector = IsolationForest(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(IsolationForest(5)) == "IsolationForest(window_size=5)"
        assert str(IsolationForest('fft')) == "IsolationForest(window_size='fft')"
        assert str(IsolationForest(15, 3)) == "IsolationForest(window_size=15,stride=3)"
        assert str(IsolationForest(25, n_estimators=42)) == "IsolationForest(window_size=25,n_estimators=42)"
        assert str(IsolationForest(25, max_samples=50)) == "IsolationForest(window_size=25,max_samples=50)"

    def test_n_estimators(self):
        IsolationForest(15, n_estimators=1)
        IsolationForest(15, n_estimators=2)
        IsolationForest(15, n_estimators=4)
        IsolationForest(15, n_estimators=8)
        with pytest.raises(TypeError):
            IsolationForest(15, n_estimators='6')
        with pytest.raises(TypeError):
            IsolationForest(15, n_estimators=6.0)
        with pytest.raises(TypeError):
            IsolationForest(15, n_estimators=True)
        with pytest.raises(ValueError):
            IsolationForest(15, n_estimators=0)
        with pytest.raises(ValueError):
            IsolationForest(15, n_estimators=-1)

    def test_max_samples(self):
        IsolationForest(15, max_samples=1)
        IsolationForest(15, max_samples=2)
        IsolationForest(15, max_samples=4)
        IsolationForest(15, max_samples=8)
        IsolationForest(15, max_samples=0.1)
        IsolationForest(15, max_samples=1.0)
        with pytest.raises(ValueError):
            IsolationForest(15, max_samples='6')
        with pytest.raises(TypeError):
            IsolationForest(15, max_samples=True)
        with pytest.raises(ValueError):
            IsolationForest(15, max_samples=0.0)
        with pytest.raises(ValueError):
            IsolationForest(15, max_samples=-0.1)
        with pytest.raises(ValueError):
            IsolationForest(15, max_samples=1.1)
        with pytest.raises(ValueError):
            IsolationForest(15, max_samples=-1)
        with pytest.raises(ValueError):
            IsolationForest(15, max_samples=0)

    def test_max_features(self):
        IsolationForest(15, max_features=1)
        IsolationForest(15, max_features=2)
        IsolationForest(15, max_features=4)
        IsolationForest(15, max_features=8)
        IsolationForest(15, max_features=0.1)
        IsolationForest(15, max_features=1.0)
        with pytest.raises(TypeError):
            IsolationForest(15, max_features='6')
        with pytest.raises(TypeError):
            IsolationForest(15, max_features=True)
        with pytest.raises(ValueError):
            IsolationForest(15, max_features=0.0)
        with pytest.raises(ValueError):
            IsolationForest(15, max_features=-0.1)
        with pytest.raises(ValueError):
            IsolationForest(15, max_features=1.1)
        with pytest.raises(ValueError):
            IsolationForest(15, max_features=-1)
        with pytest.raises(ValueError):
            IsolationForest(15, max_features=0)
