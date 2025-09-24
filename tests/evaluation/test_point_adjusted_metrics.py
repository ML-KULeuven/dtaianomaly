import numpy as np
import pytest

from dtaianomaly.evaluation import (
    PointAdjustedFBeta,
    PointAdjustedPrecision,
    PointAdjustedRecall,
)
from dtaianomaly.evaluation._point_adjusted_metrics import _point_adjust


class TestPointAdjusted:

    def test1(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        y_expc = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1])
        assert np.array_equal(_point_adjust(y_true, y_pred), y_expc)

    def test2(self):
        y_true = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        y_expc = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        assert np.array_equal(_point_adjust(y_true, y_pred), y_expc)

    def test3(self):
        y_true = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        y_expc = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1])
        assert np.array_equal(_point_adjust(y_true, y_pred), y_expc)


class TestPointAdjustedPrecision:

    def test(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedPrecision()
        assert metric.compute(y_true, y_pred) == pytest.approx(5 / 6)


class TestPointAdjustedRecall:

    def test(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedRecall()
        assert metric.compute(y_true, y_pred) == pytest.approx(5 / 7)


class TestPointAdjustedFBeta:

    def test(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedFBeta()
        assert metric.compute(y_true, y_pred) == pytest.approx(50 / 65)

    def test_beta_2(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedFBeta(beta=2)
        assert metric.compute(y_true, y_pred) == pytest.approx(125 / 170)

    def test_beta_0_point_5(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedFBeta(beta=0.5)
        assert metric.compute(y_true, y_pred) == pytest.approx(125 / 155)
