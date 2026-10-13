import numpy as np

from dtaianomaly.evaluation import BinaryMetric, Precision, ThresholdMetric
from dtaianomaly.thresholding import FixedCutoffThreshold


class _KwargsRecordingMetric(BinaryMetric):

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        self.received_kwargs = kwargs
        return 0.0


class TestThresholding:

    def test_piped_str(self):
        assert (
            ThresholdMetric(FixedCutoffThreshold(0.5), Precision()).piped_str()
            == "FixedCutoffThreshold(cutoff=0.5)->Precision()"
        )

    def test_forwards_kwargs_to_inner_metric(self):
        inner = _KwargsRecordingMetric()
        metric = ThresholdMetric(FixedCutoffThreshold(0.5), inner)
        metric.compute([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.2], foo="bar", alpha=0.5)
        assert inner.received_kwargs == {"foo": "bar", "alpha": 0.5}

    def test_no_kwargs_forwards_empty(self):
        inner = _KwargsRecordingMetric()
        metric = ThresholdMetric(FixedCutoffThreshold(0.5), inner)
        metric.compute([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.2])
        assert inner.received_kwargs == {}
