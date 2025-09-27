import inspect

import numpy as np
import pytest

from dtaianomaly.evaluation import Metric, Precision
from dtaianomaly.thresholding import FixedCutoffThreshold
from dtaianomaly.utils import all_classes


def initialize(cls):
    kwargs = {
        "beta": 1.0,
        "metric": Precision(),
        "thresholder": FixedCutoffThreshold(0.9),
    }
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(**filtered_kwargs)


@pytest.mark.numba
@pytest.mark.parametrize("cls", all_classes(Metric, return_names=False))
class TestMetrics:

    def test_non_numeric_y_true(self, cls):
        metric = initialize(cls)
        y_true = ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes"]
        y_pred = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_non_binary_y_true(self, cls):
        metric = initialize(cls)
        y_true = [0.1, 0.9, 0.5, 0.3, 0.9, 0.1, 0.2, 0.2, 0.0]
        y_pred = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_non_numeric_y_pred(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes"]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_different_size(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 0, 1, 0, 1, 0, 1, 0]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_between_0_and_1(self, cls):
        metric = initialize(cls)
        rng = np.random.default_rng()
        y_true = rng.choice([0, 1], size=1000, replace=True)
        y_pred = rng.choice([0, 1], size=1000, replace=True)
        score = metric.compute(y_true, y_pred)
        assert 0 <= score <= 1 or np.isnan(score)
