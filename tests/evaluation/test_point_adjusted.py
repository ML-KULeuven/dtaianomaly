import numpy as np
import pytest

from dtaianomaly.evaluation.point_adjusted_binary_metrics import (
    PointAdjusted,
    PointAdjustedFBeta,
    PointAdjustedPrecision,
    PointAdjustedRecall,
    point_adjust,
)
from dtaianomaly.evaluation.simple_proba_metrics import AreaUnderROC


class TestPointAdjusted:

    def test1(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        y_expc = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1])
        assert np.array_equal(point_adjust(y_true, y_pred), y_expc)

    def test2(self):
        y_true = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        y_expc = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        assert np.array_equal(point_adjust(y_true, y_pred), y_expc)

    def test3(self):
        y_true = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        y_expc = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1])
        assert np.array_equal(point_adjust(y_true, y_pred), y_expc)


class TestPointAdjustedPrecision:

    def test_string_metric(self):
        with pytest.raises(TypeError):
            PointAdjusted("Precision()")

    def test_proba_metric(self):
        with pytest.raises(TypeError):
            PointAdjusted(AreaUnderROC())

    def test(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedPrecision()
        assert metric.compute(y_true, y_pred) == pytest.approx(5 / 6)

    def test_str(self):
        assert str(PointAdjustedPrecision()) == "PointAdjustedPrecision()"


class TestPointAdjustedRecall:

    def test(self):
        y_true = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        metric = PointAdjustedRecall()
        assert metric.compute(y_true, y_pred) == pytest.approx(5 / 7)

    def test_str(self):
        assert str(PointAdjustedRecall()) == "PointAdjustedRecall()"


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

    def test_str(self):
        assert str(PointAdjustedFBeta()) == "PointAdjustedFBeta()"
        assert str(PointAdjustedFBeta(beta=0.5)) == "PointAdjustedFBeta(beta=0.5)"
        assert str(PointAdjustedFBeta(beta=2)) == "PointAdjustedFBeta(beta=2)"
