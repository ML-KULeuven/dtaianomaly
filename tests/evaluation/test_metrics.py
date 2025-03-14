
import pytest
import numpy as np
from dtaianomaly.evaluation import *
from dtaianomaly.thresholding import FixedCutoff

binary_metrics = [Precision(), Recall(), FBeta(), PointAdjustedPrecision(), PointAdjustedRecall(), PointAdjustedFBeta()]
proba_metrics = [
    AreaUnderROC(), AreaUnderPR(), ThresholdMetric(FixedCutoff(0.5), Precision()), BestThresholdMetric(Precision()),
    RangeAreaUnderPR(), RangeAreaUnderROC(), VolumeUnderPR(max_samples=20, max_buffer_size=250), VolumeUnderROC(max_samples=20, max_buffer_size=250),
]


@pytest.mark.parametrize('metric', binary_metrics + proba_metrics)
class TestMetrics:

    def test_non_numeric_y_true(self, metric):
        y_true = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes']
        y_pred = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_non_binary_y_true(self, metric):
        y_true = [0.1, 0.9, 0.5, 0.3, 0.9, 0.1, 0.2, 0.2, 0.0]
        y_pred = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_non_numeric_y_pred(self, metric):
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes']
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_different_size(self, metric):
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 0, 1, 0, 1, 0, 1, 0]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_between_0_and_1(self, metric):
        print(metric)
        rng = np.random.default_rng()
        y_true = rng.choice([0, 1], size=1000, replace=True)
        y_pred = rng.choice([0, 1], size=1000, replace=True)
        assert 0 <= metric.compute(y_true, y_pred) <= 1


@pytest.mark.parametrize('metric', binary_metrics)
class TestBinaryMetrics:

    def test_non_binary_input(self, metric):
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_combined_with_metric(self, metric):
        y_true = [0,   0,   1,   0,   1,   0,   1,   0,   1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        y_pred_thresholded = [0, 0, 1, 0, 1, 0, 1, 1, 1]
        threshold_metric = ThresholdMetric(FixedCutoff(0.5), metric)
        assert metric.compute(y_true, y_pred_thresholded) == threshold_metric.compute(y_true, y_pred)


class TestThresholding:

    def test_string_thresholding(self):
        with pytest.raises(TypeError):
            ThresholdMetric('FixedCutoff(0.5)', Precision())

    def test_string_metric(self):
        with pytest.raises(TypeError):
            ThresholdMetric(FixedCutoff(0.5), 'Precision()')

    def test_proba_metric(self):
        with pytest.raises(TypeError):
            ThresholdMetric(FixedCutoff(0.5), AreaUnderROC())

    def test_str(self):
        assert str(ThresholdMetric(FixedCutoff(0.5), Precision())) == "FixedCutoff(cutoff=0.5)->Precision()"
