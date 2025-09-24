import pytest

from dtaianomaly.evaluation import FBeta, Precision, Recall


@pytest.fixture
def data():
    return [1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]


class TestPrecision:

    def test(self, data):
        metric = Precision()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == 4 / 5


class TestRecall:

    def test(self, data):
        metric = Recall()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == 4 / 6


class TestFBeta:

    def test(self, data):
        metric = FBeta()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(8 / 11)

    def test_beta_2(self, data):
        metric = FBeta(2)
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(20 / 29)

    def test_beta_0_point_5(self, data):
        metric = FBeta(0.5)
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(10 / 13)
