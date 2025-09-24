import inspect

import pytest

from dtaianomaly.evaluation import BinaryMetric, ThresholdMetric
from dtaianomaly.thresholding import FixedCutoffThreshold
from dtaianomaly.utils import all_classes


def initialize(cls):
    kwargs = {"beta": 1.0}
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(**filtered_kwargs)


@pytest.mark.parametrize("cls", all_classes(BinaryMetric, return_names=False))
class TestBinaryMetric:

    def test_non_binary_input(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_combined_with_threshold(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        y_pred_thresholded = [0, 0, 1, 0, 1, 0, 1, 1, 1]
        threshold_metric = ThresholdMetric(FixedCutoffThreshold(0.5), metric)
        assert metric.compute(y_true, y_pred_thresholded) == threshold_metric.compute(
            y_true, y_pred
        )
