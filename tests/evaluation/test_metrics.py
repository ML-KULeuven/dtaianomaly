import inspect

import numpy as np
import pytest

from dtaianomaly import utils
from dtaianomaly.evaluation import *
from dtaianomaly.thresholding import FixedCutoff

binary_metrics = utils.all_classes("binary-metric", return_names=False)
proba_metrics = utils.all_classes("proba-metric", return_names=False)


def initialize(cls):
    kwargs = {"beta": 1.0, "metric": Precision(), "thresholder": FixedCutoff(0.9)}
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(**filtered_kwargs)


@pytest.mark.parametrize("cls", binary_metrics + proba_metrics)
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


@pytest.mark.parametrize("cls", binary_metrics)
class TestBinaryMetrics:

    def test_non_binary_input(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_combined_with_metric(self, cls):
        metric = initialize(cls)
        y_true = [0, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0.2, 0.3, 0.9, 0.4, 0.6, 0.3, 0.7, 0.6, 0.9]
        y_pred_thresholded = [0, 0, 1, 0, 1, 0, 1, 1, 1]
        threshold_metric = ThresholdMetric(FixedCutoff(0.5), metric)
        assert metric.compute(y_true, y_pred_thresholded) == threshold_metric.compute(
            y_true, y_pred
        )


class TestThresholding:

    def test_string_thresholding(self):
        with pytest.raises(TypeError):
            ThresholdMetric("FixedCutoff(0.5)", Precision())

    def test_string_metric(self):
        with pytest.raises(TypeError):
            ThresholdMetric(FixedCutoff(0.5), "Precision()")

    def test_proba_metric(self):
        with pytest.raises(TypeError):
            ThresholdMetric(FixedCutoff(0.5), AreaUnderROC())

    def test_str(self):
        assert (
            str(ThresholdMetric(FixedCutoff(0.5), Precision()))
            == "FixedCutoff(cutoff=0.5)->Precision()"
        )
