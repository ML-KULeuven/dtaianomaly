import numpy as np
import pytest

from dtaianomaly.evaluation import UCRScore


class TestUCRScore:

    def test_str(self):
        assert str(UCRScore()) == "UCRScore()"
        assert str(UCRScore(tolerance=50)) == "UCRScore(tolerance=50)"

    def test_initialize_tolerance(self):
        for t in range(1, 10):
            UCRScore(t)
        with pytest.raises(TypeError):
            UCRScore("50")
        with pytest.raises(TypeError):
            UCRScore(50.0)
        with pytest.raises(TypeError):
            UCRScore(True)
        with pytest.raises(ValueError):
            UCRScore(0)
        with pytest.raises(ValueError):
            UCRScore(-1)

    def test_no_anomalies(self):
        metric = UCRScore()
        score = metric.compute(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.3, 0.1, 0.2, 0.4, 0.3, 0.3, 0.1, 0.2, 0.1, 0.0, 0.1, 0.1, 0.0],
        )
        assert np.isnan(score)

    def test_multiple_anomalies(self):
        metric = UCRScore()
        score = metric.compute(
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            [0.1, 0.3, 0.1, 0.2, 0.4, 0.3, 0.3, 0.1, 0.2, 0.1, 0.0, 0.1, 0.1, 0.0],
        )
        assert np.isnan(score)

    def test(self):
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        y_pred_base = [
            0.1,
            0.3,
            0.1,
            0.2,
            0.4,
            0.2,
            0.3,
            0.1,
            0.2,
            0.1,
            0.0,
            0.1,
            0.1,
            0.0,
        ]
        expected = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

        for i in range(len(y_true)):
            y_pred = y_pred_base.copy()
            y_pred[i] = 0.9
            assert expected[i] == UCRScore().compute(y_true, y_pred)

    def test_tolerance(self):
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        y_pred_base = [
            0.1,
            0.3,
            0.1,
            0.2,
            0.4,
            0.2,
            0.3,
            0.1,
            0.2,
            0.1,
            0.0,
            0.1,
            0.1,
            0.0,
        ]
        expected = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        expected_tolerance = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        for i in range(len(y_true)):
            y_pred = y_pred_base.copy()
            y_pred[i] = 0.9
            assert expected[i] == UCRScore().compute(y_true, y_pred)
            assert expected_tolerance[i] == UCRScore(tolerance=5).compute(
                y_true, y_pred
            )
