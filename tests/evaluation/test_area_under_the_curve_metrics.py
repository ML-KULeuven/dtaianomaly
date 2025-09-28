import pytest

from dtaianomaly.evaluation import AreaUnderPR, AreaUnderROC


@pytest.fixture
def data():
    return [1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]


@pytest.fixture
def data_proba():
    return [1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [
        0.8,
        0.2,
        0.1,
        0.6,
        0.9,
        0.2,
        0.3,
        0.6,
        0.5,
        0.4,
    ]


class TestAreaUnderROC:

    def test(self, data):
        metric = AreaUnderROC()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(0.708, rel=1e-3)

    def test_data_proba(self, data_proba):
        metric = AreaUnderROC()
        y_true, y_pred = data_proba
        assert metric.compute(y_true, y_pred) == pytest.approx(0.479, rel=1e-3)


class TestAreaUnderPR:

    def test(self, data):
        metric = AreaUnderPR()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(0.833, rel=1e-3)

    def test_data_proba(self, data_proba):
        metric = AreaUnderPR()
        y_true, y_pred = data_proba
        assert metric.compute(y_true, y_pred) == pytest.approx(0.546, rel=1e-3)
