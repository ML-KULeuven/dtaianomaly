
import pytest

from dtaianomaly.anomaly_detection import RobustRandomCutForestAnomalyDetector, Supervision


class TestRobustRandomCutForestAnomalyDetector:

    def test_supervision(self):
        detector = RobustRandomCutForestAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(RobustRandomCutForestAnomalyDetector(5)) == "RobustRandomCutForestAnomalyDetector(window_size=5)"
        assert str(RobustRandomCutForestAnomalyDetector(10, n_estimators=10)) == "RobustRandomCutForestAnomalyDetector(window_size=10,n_estimators=10)"

    def test_precision(self):
        RobustRandomCutForestAnomalyDetector(15, stride=1)
        RobustRandomCutForestAnomalyDetector(15, stride=2)
        RobustRandomCutForestAnomalyDetector(15, stride=4)
        RobustRandomCutForestAnomalyDetector(15, stride=8)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, stride='6')
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, stride=True)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, stride=0.0)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, stride=0)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, stride=-1)

    def test_online_learning(self):
        RobustRandomCutForestAnomalyDetector(15, online_learning=True)
        RobustRandomCutForestAnomalyDetector(15, online_learning=False)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, online_learning='True')
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, online_learning=1)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, online_learning=0.0)

    def test_n_estimators(self):
        RobustRandomCutForestAnomalyDetector(15, n_estimators=1)
        RobustRandomCutForestAnomalyDetector(15, n_estimators=2)
        RobustRandomCutForestAnomalyDetector(15, n_estimators=4)
        RobustRandomCutForestAnomalyDetector(15, n_estimators=8)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, n_estimators='6')
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, n_estimators=6.0)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, n_estimators=True)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, n_estimators=0)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, n_estimators=-1)

    def test_max_samples(self):
        RobustRandomCutForestAnomalyDetector(15, max_samples=1)
        RobustRandomCutForestAnomalyDetector(15, max_samples=2)
        RobustRandomCutForestAnomalyDetector(15, max_samples=4)
        RobustRandomCutForestAnomalyDetector(15, max_samples=8)
        RobustRandomCutForestAnomalyDetector(15, max_samples=0.1)
        RobustRandomCutForestAnomalyDetector(15, max_samples=1.0)
        RobustRandomCutForestAnomalyDetector(15, max_samples="auto")
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, max_samples='6')
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, max_samples=True)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, max_samples=0.0)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, max_samples=-0.1)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, max_samples=1.1)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, max_samples=-1)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, max_samples=0)

    def test_precision(self):
        RobustRandomCutForestAnomalyDetector(15, precision=1)
        RobustRandomCutForestAnomalyDetector(15, precision=2)
        RobustRandomCutForestAnomalyDetector(15, precision=4)
        RobustRandomCutForestAnomalyDetector(15, precision=8)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, precision='6')
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, precision=True)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, precision=0.0)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, precision=0)
        with pytest.raises(ValueError):
            RobustRandomCutForestAnomalyDetector(15, precision=-1)

    def test_random_state(self):
        RobustRandomCutForestAnomalyDetector(15, random_state=0)
        RobustRandomCutForestAnomalyDetector(15, random_state=-1)
        RobustRandomCutForestAnomalyDetector(15, random_state=1)
        RobustRandomCutForestAnomalyDetector(15, random_state=2)
        RobustRandomCutForestAnomalyDetector(15, random_state=4)
        RobustRandomCutForestAnomalyDetector(15, random_state=8)
        RobustRandomCutForestAnomalyDetector(15, random_state=None)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, precision='6')
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, precision=True)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, precision=0.0)
