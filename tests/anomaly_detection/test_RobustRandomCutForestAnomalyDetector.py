import numpy as np
import pytest

from dtaianomaly.anomaly_detection import (
    RobustRandomCutForestAnomalyDetector,
    Supervision,
)


class TestRobustRandomCutForestAnomalyDetector:

    def test_supervision(self):
        detector = RobustRandomCutForestAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert (
            str(RobustRandomCutForestAnomalyDetector(5))
            == "RobustRandomCutForestAnomalyDetector(window_size=5)"
        )
        assert (
            str(RobustRandomCutForestAnomalyDetector(10, n_estimators=10))
            == "RobustRandomCutForestAnomalyDetector(window_size=10,n_estimators=10)"
        )

    def test_stride(self):
        RobustRandomCutForestAnomalyDetector(15, stride=1)
        RobustRandomCutForestAnomalyDetector(15, stride=2)
        RobustRandomCutForestAnomalyDetector(15, stride=4)
        RobustRandomCutForestAnomalyDetector(15, stride=8)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, stride="6")
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
            RobustRandomCutForestAnomalyDetector(15, online_learning="True")
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
            RobustRandomCutForestAnomalyDetector(15, n_estimators="6")
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
            RobustRandomCutForestAnomalyDetector(15, max_samples="6")
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
            RobustRandomCutForestAnomalyDetector(15, precision="6")
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
            RobustRandomCutForestAnomalyDetector(15, random_state="6")
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, random_state=True)
        with pytest.raises(TypeError):
            RobustRandomCutForestAnomalyDetector(15, random_state=0.0)


def _series(n=100, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


class TestFitAndDecisionFunction:

    @pytest.mark.parametrize("stride", [2, 3, 10])
    def test_stride(self, stride):
        X = _series()
        detector = RobustRandomCutForestAnomalyDetector(
            10, stride=stride, n_estimators=3, random_state=0
        ).fit(X)
        scores = detector.decision_function(X)
        assert scores.shape == (X.shape[0],)
        assert np.all(np.isfinite(scores)) and np.all(scores > 0)

    @pytest.mark.parametrize(
        "max_samples,expected",
        [
            ("auto", 91),
            (500, 91),
            (40, 40),
            (0.5, 45),
            pytest.param(
                0.001,
                1,
                marks=pytest.mark.xfail(
                    raises=ValueError,
                    strict=True,
                    reason="RCTree cannot be built from a single window",
                ),
            ),
        ],
    )
    def test_max_samples(self, max_samples, expected):
        X = _series()
        detector = RobustRandomCutForestAnomalyDetector(
            10, max_samples=max_samples, n_estimators=2, random_state=0
        ).fit(X)
        assert detector.max_samples_ == expected
        assert np.all(np.isfinite(detector.decision_function(X)))

    def test_max_samples_strided(self):
        X = _series()
        detector = RobustRandomCutForestAnomalyDetector(
            10, stride=5, n_estimators=2, random_state=0
        ).fit(X)
        assert detector.max_samples_ == 19

    def test_trees_have_different_seeds(self):
        detector = RobustRandomCutForestAnomalyDetector(
            10, n_estimators=3, random_state=0
        ).fit(_series())
        keys = [tuple(tree.rng.get_state()[1][:5]) for tree in detector.forest_]
        assert len(set(keys)) == len(keys)

    def test_reproducible(self):
        X = _series()
        scores = [
            RobustRandomCutForestAnomalyDetector(10, n_estimators=3, random_state=0)
            .fit(X)
            .decision_function(X)
            for _ in range(2)
        ]
        assert np.array_equal(scores[0], scores[1])
