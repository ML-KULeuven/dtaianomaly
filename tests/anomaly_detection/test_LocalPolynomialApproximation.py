import pytest

from dtaianomaly.anomaly_detection import LocalPolynomialApproximation, Supervision


class TestLocalPolynomialApproximation:

    def test_supervision(self):
        detector = LocalPolynomialApproximation(16)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert (
            str(LocalPolynomialApproximation(5))
            == "LocalPolynomialApproximation(neighborhood=5)"
        )
        assert (
            str(LocalPolynomialApproximation(15, 3))
            == "LocalPolynomialApproximation(neighborhood=15,power=3)"
        )
        assert (
            str(LocalPolynomialApproximation(25, buffer=42))
            == "LocalPolynomialApproximation(neighborhood=25,buffer=42)"
        )

    def test_neighborhood(self):
        LocalPolynomialApproximation(15)
        LocalPolynomialApproximation(64)
        LocalPolynomialApproximation(32)
        LocalPolynomialApproximation(2)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(True)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15.0)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation("15")
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(-1)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(0)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(1)

    def test_power(self):
        for p in range(1, 10):
            LocalPolynomialApproximation(15, p)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, "6")
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, 6.0)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, True)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(15, 0)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(15, -1)

    def test_normalize_variance(self):
        LocalPolynomialApproximation(15, normalize_variance=True)
        LocalPolynomialApproximation(15, normalize_variance=False)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, normalize_variance="True")
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, normalize_variance=1)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, normalize_variance=0.0)

    def test_buffer(self):
        LocalPolynomialApproximation(15, buffer=3)
        LocalPolynomialApproximation(15, buffer=16)
        LocalPolynomialApproximation(15, buffer=32)
        LocalPolynomialApproximation(15, buffer=128)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, buffer="18")
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, buffer=True)
        with pytest.raises(TypeError):
            LocalPolynomialApproximation(15, buffer=16.0)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(15, buffer=0)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(15, buffer=-1)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(15, buffer=1)
        with pytest.raises(ValueError):
            LocalPolynomialApproximation(15, buffer=2)
