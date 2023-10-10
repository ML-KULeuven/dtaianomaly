
import pytest
import numpy as np

from dtaianomaly.evaluation.classification_metrics import precision, recall, f1
from tests.evaluation.TestEvaluationUtil import TestEvaluationUtil


class TestClassificationMetrics:

    ground_truth: np.array = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    predicted: np.array = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    nb_reps: int = 100
    length_synthetic: int = 1000

    def test_precision(self):
        assert np.count_nonzero(self.predicted) == 8
        assert precision(self.ground_truth, self.predicted) == pytest.approx(5 / 8)

    def test_recall(self):
        assert np.count_nonzero(self.ground_truth) == 8
        assert recall(self.ground_truth, self.predicted) == pytest.approx(5 / 8)

    def test_f1(self):
        assert f1(self.ground_truth, self.predicted) == pytest.approx(5 / 8)

    def test_util(self):
        assert TestEvaluationUtil.true_positive(self.ground_truth, self.predicted) == 5
        assert TestEvaluationUtil.false_positive(self.ground_truth, self.predicted) == 3
        assert TestEvaluationUtil.false_negative(self.ground_truth, self.predicted) == 3

    def test_precision_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fp = TestEvaluationUtil.false_positive(ground_truth, predicted)
            assert precision(ground_truth, predicted) == pytest.approx(tp / (tp + fp))

    def test_recall_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fn = TestEvaluationUtil.false_negative(ground_truth, predicted)
            assert recall(ground_truth, predicted) == pytest.approx(tp / (tp + fn))

    def test_f1_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fp = TestEvaluationUtil.false_positive(ground_truth, predicted)
            fn = TestEvaluationUtil.false_negative(ground_truth, predicted)
            assert f1(ground_truth, predicted) == pytest.approx(2*tp / (2*tp + fp + fn))
