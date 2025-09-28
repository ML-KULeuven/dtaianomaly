import inspect

import pytest

from dtaianomaly.evaluation import Precision, ProbaMetric
from dtaianomaly.thresholding import FixedCutoffThreshold
from dtaianomaly.utils import all_classes


def initialize(cls):
    kwargs = {"metric": Precision(), "thresholder": FixedCutoffThreshold(0.9)}
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(**filtered_kwargs)


@pytest.mark.numba
@pytest.mark.parametrize("cls", all_classes(ProbaMetric, return_names=False))
class TestProbaMetric:

    def test_non_binary_input(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        metric.compute(y_true, y_pred)  # Not an issue
