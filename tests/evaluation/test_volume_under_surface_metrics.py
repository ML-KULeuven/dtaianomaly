import numpy as np
import pytest

from dtaianomaly.evaluation import (
    RangeAreaUnderPR,
    RangeAreaUnderROC,
    VolumeUnderPR,
    VolumeUnderROC,
)


@pytest.fixture
def y_true():
    y = np.zeros(200)
    y[10:20] = 1
    y[28:33] = 1
    y[110:120] = 1
    return y


@pytest.fixture
def y_pred():
    y = np.random.default_rng(41).random(200) * 0.5
    y[16:22] = 1
    y[33:38] = 1
    y[160:170] = 1
    return y


def _small_y_true():
    return np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def _small_y_pred():
    # Perfect ranking: every anomaly scores strictly higher than every normal sample.
    return np.array([0.05, 0.2, 1.0, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])


@pytest.mark.numba
class TestVUSMetrics:

    def test_range_pr_auc_fixture(self, y_true, y_pred):
        result = RangeAreaUnderPR().compute(y_true, y_pred)
        assert result == pytest.approx(0.3973939890)

    def test_range_roc_auc_fixture(self, y_true, y_pred):
        result = RangeAreaUnderROC().compute(y_true, y_pred)
        assert result == pytest.approx(0.7991799044)

    def test_edge_case_existence_reward(self, y_true, y_pred):
        result = RangeAreaUnderPR(buffer_size=4).compute(y_true, y_pred)
        assert result == pytest.approx(0.2541383324)

        result = RangeAreaUnderROC(buffer_size=4).compute(y_true, y_pred)
        assert result == pytest.approx(0.6801506893)

    def test_range_pr_volume_fixture(self, y_true, y_pred):
        result = VolumeUnderPR(max_buffer_size=200).compute(y_true, y_pred)
        assert result == pytest.approx(0.7289826681)

    def test_range_roc_volume_fixture(self, y_true, y_pred):
        result = VolumeUnderROC(max_buffer_size=200).compute(y_true, y_pred)
        assert result == pytest.approx(0.9389229978)

    def test_range_pr_volume_default_max_buffer_size(self, y_true, y_pred):
        result = VolumeUnderPR().compute(y_true, y_pred)
        assert result == pytest.approx(0.5415335431)

    def test_range_roc_volume_default_max_buffer_size(self, y_true, y_pred):
        result = VolumeUnderROC().compute(y_true, y_pred)
        assert result == pytest.approx(0.8881003620)

    def test_range_pr_auc_small(self):
        result = RangeAreaUnderPR().compute(_small_y_true(), _small_y_pred())
        assert result == pytest.approx(1.0)

    def test_range_roc_auc_small(self):
        result = RangeAreaUnderROC().compute(_small_y_true(), _small_y_pred())
        assert result == pytest.approx(1.0)

    def test_range_pr_volume_small(self):
        result = VolumeUnderPR(max_buffer_size=200).compute(
            _small_y_true(), _small_y_pred()
        )
        assert result == pytest.approx(1.0)

    def test_range_roc_volume_small(self):
        result = VolumeUnderROC(max_buffer_size=200).compute(
            _small_y_true(), _small_y_pred()
        )
        assert result == pytest.approx(1.0)
