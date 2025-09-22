import pytest

from dtaianomaly.evaluation import AreaUnderPR, AreaUnderROC, FBeta, Precision, Recall


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


class TestPrecision:

    def test(self, data):
        metric = Precision()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == 4 / 5

    def test_str(self):
        assert str(Precision()) == "Precision()"


class TestRecall:

    def test(self, data):
        metric = Recall()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == 4 / 6

    def test_str(self):
        assert str(Recall()) == "Recall()"


class TestFBeta:

    def test_default_beta(self):
        f_beta = FBeta()
        assert f_beta.beta == 1.0

    def test_string_beta(self):
        with pytest.raises(TypeError):
            FBeta("1.0")

    def test_bool_beta(self):
        with pytest.raises(TypeError):
            FBeta(True)

    def test_zero_beta(self):
        with pytest.raises(ValueError):
            FBeta(0.0)

    def test_negative_beta(self):
        with pytest.raises(ValueError):
            FBeta(-1.0)

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

    def test_str(self):
        assert str(FBeta()) == "FBeta()"
        assert str(FBeta(2)) == "FBeta(beta=2)"
        assert str(FBeta(0.5)) == "FBeta(beta=0.5)"


class TestAreaUnderROC:

    def test(self, data):
        metric = AreaUnderROC()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(0.708, rel=1e-3)

    def test_data_proba(self, data_proba):
        metric = AreaUnderROC()
        y_true, y_pred = data_proba
        assert metric.compute(y_true, y_pred) == pytest.approx(0.479, rel=1e-3)

    def test_str(self):
        assert str(AreaUnderROC()) == "AreaUnderROC()"


class TestAreaUnderPR:

    def test(self, data):
        metric = AreaUnderPR()
        y_true, y_pred = data
        assert metric.compute(y_true, y_pred) == pytest.approx(0.833, rel=1e-3)

    def test_data_proba(self, data_proba):
        metric = AreaUnderPR()
        y_true, y_pred = data_proba
        assert metric.compute(y_true, y_pred) == pytest.approx(0.546, rel=1e-3)

    def test_str(self):
        assert str(AreaUnderPR()) == "AreaUnderPR()"
