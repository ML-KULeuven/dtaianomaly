import numpy as np
import pytest

from dtaianomaly.thresholding import TopNThreshold


class TestTopNThreshold:

    def test(self):
        ground_truth = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.8])
        thresholder = TopNThreshold(n=2)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_too_large_n(self):
        thresholder = TopNThreshold(5)
        with pytest.raises(ValueError):
            thresholder.threshold(np.array([0.0, 0.9, 1.0]))

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid_n(self, n):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            TopNThreshold(n)

    def test_n_one_flags_only_max(self):
        thresholder = TopNThreshold(1)
        labels = thresholder.threshold(np.array([0.1, 0.9, 0.3, 0.5]))
        assert np.array_equal(labels, np.array([0, 1, 0, 0]))
