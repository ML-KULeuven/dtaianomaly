import numpy as np
import pytest

from dtaianomaly.thresholding import ContaminationRate, FixedCutoff, TopN


class TestFixedCutoffThresholding:

    def test_cutoff(self):
        ground_truth = np.array([1, 0, 1, 1])
        scores = np.array([1.0, 0.0, 0.5, 0.3])
        thresholder = FixedCutoff(cutoff=0.3)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_invalid_cutoff(self):
        with pytest.raises(TypeError):
            FixedCutoff(1)

    def test_invalid_scores(self):
        thresholder = FixedCutoff(0.9)
        with pytest.raises(ValueError):
            thresholder.threshold([0.0, "0.9", 1.0])

    def test_str(self):
        assert str(FixedCutoff(0.9)) == "FixedCutoff(cutoff=0.9)"


class TestContaminationRateThresholding:

    def test(self):
        ground_truth = np.array([0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6])
        thresholder = ContaminationRate(contamination_rate=0.25)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_all_same_scores(self):
        ground_truth = np.array([1, 1, 1, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        thresholder = ContaminationRate(contamination_rate=0.25)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_non_clean(self):
        ground_truth = np.array([0, 0, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.4, 0.6, 0.3])
        thresholder = ContaminationRate(contamination_rate=0.25)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_string_contamination(self):
        with pytest.raises(TypeError):
            ContaminationRate(contamination_rate="something else")

    def test_bool_contamination(self):
        with pytest.raises(TypeError):
            ContaminationRate(contamination_rate=True)

    def test_negative_contamination(self):
        with pytest.raises(ValueError):
            ContaminationRate(contamination_rate=-0.1)

    def test_positive_contamination(self):
        with pytest.raises(ValueError):
            ContaminationRate(contamination_rate=2.0)

    def test_invalid_scores(self):
        thresholder = ContaminationRate(0.1)
        with pytest.raises(ValueError):
            thresholder.threshold([0.0, "0.9", 1.0])

    def test_str(self):
        assert (
            str(ContaminationRate(0.1)) == "ContaminationRate(contamination_rate=0.1)"
        )


class TestTopN:

    def test(self):
        ground_truth = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.8])
        thresholder = TopN(n=2)
        assert np.array_equal(ground_truth, thresholder.threshold(scores))

    def test_string_topn(self):
        with pytest.raises(TypeError):
            TopN(n="something else")

    def test_bool_topn(self):
        with pytest.raises(TypeError):
            TopN(n=True)

    def test_negative_topn(self):
        with pytest.raises(ValueError):
            TopN(n=-1)

    def test_invalid_scores(self):
        thresholder = TopN(2)
        with pytest.raises(ValueError):
            thresholder.threshold([0.0, "0.9", 1.0])

    def test_too_large_n(self):
        thresholder = TopN(5)
        with pytest.raises(ValueError):
            thresholder.threshold(np.array([0.0, 0.9, 1.0]))

    def test_str(self):
        assert str(TopN(5)) == "TopN(n=5)"
