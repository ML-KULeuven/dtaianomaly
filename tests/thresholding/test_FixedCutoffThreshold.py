import numpy as np

from dtaianomaly.thresholding import FixedCutoffThreshold


class TestFixedCutoffThreshold:

    def test_cutoff(self):
        ground_truth = np.array([1, 0, 1, 1])
        scores = np.array([1.0, 0.0, 0.5, 0.3])
        thresholder = FixedCutoffThreshold(cutoff=0.3)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))
