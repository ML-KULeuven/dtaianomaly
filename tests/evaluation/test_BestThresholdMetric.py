import numpy as np
import pytest

from dtaianomaly.evaluation import (
    BestThresholdMetric,
    BinaryMetric,
    FBeta,
    Precision,
    Recall,
)


class _KwargsRecordingMetric(BinaryMetric):

    def __init__(self):
        self.received_kwargs = []

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        self.received_kwargs.append(kwargs)
        return 0.0


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
        assert metric.threshold_ == pytest.approx(0.2)
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
        assert metric.threshold_ == pytest.approx(13 / 30)
        assert metric.thresholds_.shape == (4,)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert metric.thresholds_ == pytest.approx([0.2, 13 / 30, 2 / 3, 0.9])

    def test_fbeta_2_subset_thresholds_quantile(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(
            FBeta(), max_nb_thresholds=4, binning_strategy="quantile"
        )
        assert metric.compute(y_true, y_pred) == pytest.approx(0.8333333, abs=1e-5)
        assert metric.threshold_ == pytest.approx(0.4)
        assert metric.thresholds_.shape == (4,)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert metric.thresholds_ == pytest.approx([0.25, 0.4, 0.65, 0.75])

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
        assert metric.threshold_ == pytest.approx(0.2)
        assert metric.thresholds_.shape == (2,)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert np.array_equal(metric.thresholds_, np.array([0.2, 0.9]))

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
            (1, [0.1]),
            (2, [0.1, 0.9]),
            (3, [0.1, 0.5, 0.9]),
            (4, [0.1, 0.3667, 0.6333, 0.9]),
            (5, [0.1, 0.3, 0.5, 0.7, 0.9]),
            (11, [0.1, 0.18, 0.26, 0.34, 0.42, 0.5, 0.58, 0.66, 0.74, 0.82, 0.9]),
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

    def test_uniform_thresholds_respect_narrow_score_range(self):
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=3)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]), np.array([0.4, 0.45, 0.55, 0.6, 0.5, 0.42])
        )
        assert metric.thresholds_ == pytest.approx([0.4, 0.5, 0.6])

    def test_uniform_thresholds_respect_given_threshold_range(self):
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=3)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]),
            np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.1]),
            thresholds=np.array([0.5, 0.3, 0.7, 0.4]),
        )
        assert metric.thresholds_ == pytest.approx([0.3, 0.5, 0.7])

    def test_only_few_thresholds(self):
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=100)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]), np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
        )
        assert metric.thresholds_ == pytest.approx([0.1, 0.5], abs=1e-3)

    @pytest.mark.parametrize(
        "y_pred",
        [
            np.array([-3.0, -2.5, -0.5, -0.2, -2.0, -2.8]),
            np.array([10.0, 12.0, 45.0, 50.0, 20.0, 11.0]),
            np.array([-5.0, -1.0, 3.0, 7.5, 0.5, -4.0]),
        ],
    )
    def test_first_threshold_is_minimum_score(self, y_pred):
        metric = BestThresholdMetric(FBeta())
        metric.compute(np.array([0, 0, 1, 1, 0, 0]), y_pred)
        assert metric.thresholds_[0] == y_pred.min()

    @pytest.mark.parametrize(
        "y_pred",
        [
            np.array([-3.0, -2.5, -0.5, -0.2, -2.0, -2.8]),
            np.array([10.0, 12.0, 45.0, 50.0, 20.0, 11.0]),
            np.array([-5.0, -1.0, 3.0, 7.5, 0.5, -4.0]),
        ],
    )
    def test_recall_reaches_one_outside_unit_interval(self, y_pred):
        metric = BestThresholdMetric(Recall())
        assert metric.compute(np.array([0, 0, 1, 1, 0, 0]), y_pred) == pytest.approx(
            1.0
        )

    @pytest.mark.parametrize(
        "y_pred",
        [
            np.array([-3.0, -2.5, -0.5, -0.2, -2.0, -2.8]),
            np.array([10.0, 12.0, 45.0, 50.0, 20.0, 11.0]),
        ],
    )
    def test_fbeta_separable_outside_unit_interval(self, y_pred):
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(np.array([0, 0, 1, 1, 0, 0]), y_pred) == pytest.approx(
            1.0
        )
        assert y_pred[[0, 1, 4, 5]].max() < metric.threshold_ <= y_pred[[2, 3]].min()

    def test_forwards_kwargs_to_inner_metric_for_every_threshold(self):
        inner = _KwargsRecordingMetric()
        metric = BestThresholdMetric(inner)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]),
            np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.1]),
            foo="bar",
        )
        assert len(inner.received_kwargs) == len(metric.thresholds_)
        assert all(kwargs == {"foo": "bar"} for kwargs in inner.received_kwargs)

    def test_does_not_forward_given_thresholds_to_inner_metric(self):
        inner = _KwargsRecordingMetric()
        metric = BestThresholdMetric(inner)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]),
            np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.1]),
            thresholds=np.array([0.15, 0.5]),
            foo="bar",
        )
        assert len(inner.received_kwargs) == 2
        assert all(kwargs == {"foo": "bar"} for kwargs in inner.received_kwargs)

    def test_no_kwargs_forwards_empty(self):
        inner = _KwargsRecordingMetric()
        metric = BestThresholdMetric(inner)
        metric.compute(
            np.array([0, 0, 1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.1])
        )
        assert all(kwargs == {} for kwargs in inner.received_kwargs)
