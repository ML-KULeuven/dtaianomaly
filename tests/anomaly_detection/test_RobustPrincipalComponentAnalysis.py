import numpy as np
import pytest

from dtaianomaly.anomaly_detection import RobustPrincipalComponentAnalysis, Supervision
from dtaianomaly.windowing import reverse_sliding_window, sliding_window


class TestRobustPrincipalComponentAnalysis:

    def test_supervision(self):
        detector = RobustPrincipalComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_initialize_non_int_window_size(self):
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(window_size=True)
        with pytest.raises(ValueError):
            RobustPrincipalComponentAnalysis(window_size="a string")
        RobustPrincipalComponentAnalysis(5)  # Doesn't raise an error

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            RobustPrincipalComponentAnalysis(window_size=0)
        RobustPrincipalComponentAnalysis(5)  # Doesn't raise an error

    def test_valid_window_size(self):
        RobustPrincipalComponentAnalysis(1)
        RobustPrincipalComponentAnalysis(10)
        RobustPrincipalComponentAnalysis(100)
        RobustPrincipalComponentAnalysis("fft")

    def test_initialize_non_int_max_iter(self):
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(1, max_iter=True)
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(1, max_iter="a string")
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(1, max_iter=0.05)
        RobustPrincipalComponentAnalysis(5)  # Doesn't raise an error

    def test_initialize_too_small_max_iter(self):
        with pytest.raises(ValueError):
            RobustPrincipalComponentAnalysis(1, max_iter=0)
        RobustPrincipalComponentAnalysis(1)  # Doesn't raise an error

    def test_str(self):
        assert (
            str(RobustPrincipalComponentAnalysis(1))
            == "RobustPrincipalComponentAnalysis(window_size=1)"
        )
        assert (
            str(RobustPrincipalComponentAnalysis(1, max_iter=50))
            == "RobustPrincipalComponentAnalysis(window_size=1,max_iter=50)"
        )


def _sine_with_spikes():
    X = np.sin(np.arange(500) / 5)
    spikes = np.array([100, 250, 400])
    X[spikes] += 5
    return X, spikes


class TestRobustPrincipalComponentAnalysisLowRank:

    def test_keeps_fewer_components_than_window_size(self):
        X, _ = _sine_with_spikes()
        detector = RobustPrincipalComponentAnalysis(16).fit(X)
        assert detector.pca_.n_components_ < detector.window_size_

    def test_scores_are_not_summed_absolute_input(self):
        X, _ = _sine_with_spikes()
        detector = RobustPrincipalComponentAnalysis(16).fit(X)
        windows = sliding_window(X, detector.window_size_, 1)
        trivial = reverse_sliding_window(
            np.abs(windows).sum(axis=1), detector.window_size_, 1, X.shape[0]
        )
        assert not np.allclose(detector.decision_function(X), trivial)

    def test_spikes_have_higher_scores(self):
        X, spikes = _sine_with_spikes()
        y_pred = RobustPrincipalComponentAnalysis(16).fit(X).decision_function(X)
        normal = np.ones(X.shape[0], dtype=bool)
        for spike in spikes:
            normal[spike - 16 : spike + 16] = False
        assert y_pred[spikes].min() > y_pred[normal].max()
