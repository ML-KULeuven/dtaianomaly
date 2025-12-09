import numpy as np
import pytest

from dtaianomaly.evaluation import BestThresholdMetric, FBeta, Precision, Recall


class TestBestThresholdMetric:

    def test_precision(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.6])
        # Sorted scores: [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(Precision())
        assert metric.compute(y_true, y_pred) == pytest.approx(1.0)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_recall(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.6])
        # Sorted scores: [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(Recall())
        assert metric.compute(y_true, y_pred) == pytest.approx(1.0)
        assert metric.threshold_ == pytest.approx(0.0)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.6])
        # Sorted scores: [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(y_true, y_pred) == pytest.approx(1.0)
        assert metric.threshold_ == pytest.approx(0.65)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(y_true, y_pred) == pytest.approx(0.8333333, abs=1e-5)
        assert metric.threshold_ == pytest.approx(0.4)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2_subset_thresholds_uniform(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=4)
        assert metric.compute(y_true, y_pred) == pytest.approx(0.83333333333)
        assert metric.threshold_ == pytest.approx(1 / 3)
        assert metric.thresholds_.shape == (4,)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2_subset_thresholds_quantile(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(
            FBeta(), max_nb_thresholds=4, binning_strategy="quantile"
        )
        assert metric.compute(y_true, y_pred) == pytest.approx(0.75)
        assert metric.threshold_ == pytest.approx(0.65)
        assert metric.thresholds_.shape == (4,)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2_given_thresholds(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = np.array([0.2, 0.3, 0.65, 0.85, 0.9])
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(y_true, y_pred, thresholds=thresholds) == pytest.approx(
            0.75
        )
        assert metric.threshold_ == pytest.approx(0.65)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert np.array_equal(metric.thresholds_, thresholds)

    def test_fbeta_2_given_thresholds_and_subset_uniform(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = np.array([0.2, 0.3, 0.65, 0.85, 0.9])
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=2)
        assert metric.compute(y_true, y_pred, thresholds=thresholds) == pytest.approx(
            0.625
        )
        assert metric.threshold_ == pytest.approx(0.0)
        assert metric.thresholds_.shape == (2,)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert np.array_equal(metric.thresholds_, np.array([0.0, 1.0]))

    def test_fbeta_2_given_thresholds_and_subset_quantile(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = np.array([0.2, 0.3, 0.65, 0.85, 0.9])
        metric = BestThresholdMetric(
            FBeta(), max_nb_thresholds=2, binning_strategy="quantile"
        )
        assert metric.compute(y_true, y_pred, thresholds=thresholds) == pytest.approx(
            2 / 3
        )
        assert metric.threshold_ == pytest.approx(0.3)
        assert metric.thresholds_.shape == (2,)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert np.array_equal(metric.thresholds_, np.array([0.3, 0.85]))

    @pytest.mark.parametrize(
        "max_nb_thresholds,thresholds",
        [
            (1, [0.0]),
            (2, [0.0, 1.0]),
            (3, [0.0, 0.5, 1.0]),
            (4, [0.0, 0.3333, 0.6666, 1.0]),
            (5, [0.0, 0.25, 0.5, 0.75, 1.0]),
            (11, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ],
    )
    def test_uniform_thresholds(self, max_nb_thresholds, thresholds):
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=max_nb_thresholds)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array(
                [
                    0.1,
                    0.4,
                    0.8,
                    0.9,
                    0.5,
                    0.39,
                    0.38,
                    0.37,
                    0.36,
                    0.35,
                    0.34,
                    0.33,
                    0.32,
                ]
            ),
        )
        assert metric.thresholds_ == pytest.approx(thresholds, abs=1e-3)

    def test_only_few_thresholds(self):
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=100)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]), np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
        )
        assert metric.thresholds_ == pytest.approx([0, 0.5, 1.0], abs=1e-3)
