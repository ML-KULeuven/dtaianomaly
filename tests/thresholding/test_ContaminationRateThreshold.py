import numpy as np

from dtaianomaly.thresholding import ContaminationRateThreshold


class TestContaminationRateThreshold:

    def test(self):
        ground_truth = np.array([0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6])
        thresholder = ContaminationRateThreshold(contamination_rate=0.25)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_all_same_scores(self):
        ground_truth = np.array([1, 1, 1, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        thresholder = ContaminationRateThreshold(contamination_rate=0.25)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_non_clean(self):
        ground_truth = np.array([0, 0, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.4, 0.6, 0.3])
        thresholder = ContaminationRateThreshold(contamination_rate=0.25)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))
