import numpy as np
import pytest

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

    @pytest.mark.parametrize(
        "scores",
        [
            np.array([0.1, 0.2, 0.3, 0.6]),
            np.array([0.1, 0.6, 0.6, 0.6]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ],
    )
    def test_zero_rate_flags_nothing(self, scores):
        thresholder = ContaminationRateThreshold(contamination_rate=0.0)
        assert np.array_equal(thresholder.threshold(scores), np.zeros(4))

    def test_full_rate_flags_everything(self):
        scores = np.array([0.1, 0.2, 0.3, 0.6])
        thresholder = ContaminationRateThreshold(contamination_rate=1.0)
        assert np.array_equal(thresholder.threshold(scores), np.ones(4))

    @pytest.mark.parametrize("contamination_rate", [0.0, 0.25, 1.0])
    def test_raw_labels_are_int8(self, contamination_rate):
        thresholder = ContaminationRateThreshold(contamination_rate=contamination_rate)
        labels = thresholder._threshold(np.array([0.1, 0.2, 0.3, 0.6]))
        assert labels.dtype == np.int8
