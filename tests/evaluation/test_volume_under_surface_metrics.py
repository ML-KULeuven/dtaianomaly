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


class TestVUSMetrics:

    def test_range_pr_auc_compat(self, y_true, y_pred):
        result = RangeAreaUnderPR(compatibility_mode=True).compute(y_true, y_pred)
        assert result == pytest.approx(0.3737854660)

    def test_range_roc_auc_compat(self, y_true, y_pred):
        result = RangeAreaUnderROC(compatibility_mode=True).compute(y_true, y_pred)
        assert result == pytest.approx(0.7108527197)

    def test_edge_case_existence_reward_compat(self, y_true, y_pred):
        result = RangeAreaUnderPR(compatibility_mode=True, buffer_size=4).compute(
            y_true, y_pred
        )
        assert result == pytest.approx(0.2506464391)

        result = RangeAreaUnderROC(compatibility_mode=True, buffer_size=4).compute(
            y_true, y_pred
        )
        assert result == pytest.approx(0.6143220816)

    def test_range_pr_volume_compat(self, y_true, y_pred):
        result = VolumeUnderPR(max_buffer_size=200, compatibility_mode=True).compute(
            y_true, y_pred
        )
        assert result == pytest.approx(0.7493254559)

    def test_range_roc_volume_compat(self, y_true, y_pred):
        result = VolumeUnderROC(max_buffer_size=200, compatibility_mode=True).compute(
            y_true, y_pred
        )
        assert result == pytest.approx(0.8763382130)

    def test_range_pr_auc(self):
        y_pred = np.array([0.05, 0.2, 1.0, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeAreaUnderPR().compute(y_true, y_pred)
        assert result == pytest.approx(0.9636, 1e-4)

    def test_range_roc_auc(self):
        y_pred = np.array([0.05, 0.2, 1.0, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeAreaUnderROC().compute(y_true, y_pred)
        assert result == pytest.approx(0.9653, 1e-4)

    def test_range_pr_volume(self):
        y_pred = np.array([0.05, 0.2, 1.0, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = VolumeUnderPR(max_buffer_size=200).compute(y_true, y_pred)
        assert result == pytest.approx(0.9937, 1e-4)

    def test_range_roc_volume(self):
        y_pred = np.array([0.05, 0.2, 1.0, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = VolumeUnderROC(max_buffer_size=200).compute(y_true, y_pred)
        assert result == pytest.approx(0.9904, 1e-4)
