
import pytest
import numpy as np
from dtaianomaly.evaluation.volume_under_surface import RangeAreaUnderROC, RangeAreaUnderPR, VolumeUnderROC, VolumeUnderPR


class TestInitialization:

    @pytest.mark.parametrize('metric', [RangeAreaUnderROC, RangeAreaUnderPR])
    @pytest.mark.parametrize('buffer_size', [-1, 0])
    def test_invalid_buffer_size(self, metric, buffer_size):
        with pytest.raises(ValueError):
            metric(buffer_size=buffer_size)

    @pytest.mark.parametrize('metric', [RangeAreaUnderROC, RangeAreaUnderPR])
    @pytest.mark.parametrize('buffer_size', [True, 10.0, '10'])
    def test_invalid_buffer_size_type(self, metric, buffer_size):
        with pytest.raises(TypeError):
            metric(buffer_size=buffer_size)

    @pytest.mark.parametrize('metric', [VolumeUnderROC, VolumeUnderPR])
    @pytest.mark.parametrize('max_buffer_size', [-1, 0])
    def test_invalid_max_buffer_size(self, metric, max_buffer_size):
        with pytest.raises(ValueError):
            metric(max_buffer_size=max_buffer_size)

    @pytest.mark.parametrize('metric', [VolumeUnderROC, VolumeUnderPR])
    @pytest.mark.parametrize('max_buffer_size', [True, 10.0, '10'])
    def test_invalid_max_buffer_size_type(self, metric, max_buffer_size):
        with pytest.raises(TypeError):
            metric(max_buffer_size=max_buffer_size)

    @pytest.mark.parametrize('metric', [RangeAreaUnderROC, RangeAreaUnderPR, VolumeUnderROC, VolumeUnderPR])
    @pytest.mark.parametrize('compatibility_mode', [0, 0.0, ''])
    def test_invalid_compatibility_mode_type(self, metric, compatibility_mode):
        with pytest.raises(TypeError):
            metric(compatibility_mode=compatibility_mode)

    @pytest.mark.parametrize('metric', [RangeAreaUnderROC, RangeAreaUnderPR, VolumeUnderROC, VolumeUnderPR])
    @pytest.mark.parametrize('max_samples', [-1, 0])
    def test_invalid_max_samples(self, metric, max_samples):
        with pytest.raises(ValueError):
            metric(max_samples=max_samples)

    @pytest.mark.parametrize('metric', [RangeAreaUnderROC, RangeAreaUnderPR, VolumeUnderROC, VolumeUnderPR])
    @pytest.mark.parametrize('max_samples', [True, 50.0, '100'])
    def test_invalid_max_samples_type(self, metric, max_samples):
        with pytest.raises(TypeError):
            metric(max_samples=max_samples)


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
        result = RangeAreaUnderPR(compatibility_mode=True, buffer_size=4).compute(y_true, y_pred)
        assert result == pytest.approx(0.2506464391)

        result = RangeAreaUnderROC(compatibility_mode=True, buffer_size=4).compute(y_true, y_pred)
        assert result == pytest.approx(0.6143220816)

    def test_range_pr_volume_compat(self, y_true, y_pred):
        result = VolumeUnderPR(max_buffer_size=200, compatibility_mode=True).compute(y_true, y_pred)
        assert result == pytest.approx(0.7493254559)

    def test_range_roc_volume_compat(self, y_true, y_pred):
        result = VolumeUnderROC(max_buffer_size=200, compatibility_mode=True).compute(y_true, y_pred)
        assert result == pytest.approx(0.8763382130 )

    def test_range_pr_auc(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeAreaUnderPR().compute(y_true, y_pred)
        assert result == pytest.approx(0.9636, 1e-4)

    def test_range_roc_auc(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeAreaUnderROC().compute(y_true, y_pred)
        assert result == pytest.approx(0.9653, 1e-4)

    def test_range_pr_volume(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = VolumeUnderPR(max_buffer_size=200).compute(y_true, y_pred)
        assert result == pytest.approx(0.9937, 1e-4)

    def test_range_roc_volume(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = VolumeUnderROC(max_buffer_size=200).compute(y_true, y_pred)
        assert result == pytest.approx(0.9904, 1e-4)


class TestStr:

    def test_range_roc(self):
        assert str(RangeAreaUnderPR()) == "RangeAreaUnderPR()"
        assert str(RangeAreaUnderPR(buffer_size=50)) == "RangeAreaUnderPR(buffer_size=50)"
        assert str(RangeAreaUnderPR(compatibility_mode=True)) == "RangeAreaUnderPR(compatibility_mode=True)"
        assert str(RangeAreaUnderPR(buffer_size=50, max_samples=150)) == "RangeAreaUnderPR(buffer_size=50,max_samples=150)"

    def test_range_pr(self):
        assert str(RangeAreaUnderROC()) == "RangeAreaUnderROC()"
        assert str(RangeAreaUnderROC(buffer_size=50)) == "RangeAreaUnderROC(buffer_size=50)"
        assert str(RangeAreaUnderROC(compatibility_mode=True)) == "RangeAreaUnderROC(compatibility_mode=True)"
        assert str(RangeAreaUnderROC(buffer_size=50, max_samples=150)) == "RangeAreaUnderROC(buffer_size=50,max_samples=150)"

    def test_vus_roc(self):
        assert str(VolumeUnderPR()) == "VolumeUnderPR()"
        assert str(VolumeUnderPR(max_buffer_size=50)) == "VolumeUnderPR(max_buffer_size=50)"
        assert str(VolumeUnderPR(compatibility_mode=True)) == "VolumeUnderPR(compatibility_mode=True)"
        assert str(VolumeUnderPR(max_buffer_size=50, max_samples=150)) == "VolumeUnderPR(max_buffer_size=50,max_samples=150)"

    def test_vus_pr(self):
        assert str(VolumeUnderROC()) == "VolumeUnderROC()"
        assert str(VolumeUnderROC(max_buffer_size=50)) == "VolumeUnderROC(max_buffer_size=50)"
        assert str(VolumeUnderROC(compatibility_mode=True)) == "VolumeUnderROC(compatibility_mode=True)"
        assert str(VolumeUnderROC(max_buffer_size=50, max_samples=150)) == "VolumeUnderROC(max_buffer_size=50,max_samples=150)"

    @pytest.mark.parametrize('metric,string', [
        (VolumeUnderROC(), "VolumeUnderROC()"),
        (VolumeUnderPR(), "VolumeUnderPR()")
    ])
    def test_vus_roc_after_running(self, metric, string):
        # Buffer size is not None, but this shouldn't show in the string approach
        assert str(metric) == string
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        metric.compute(y_true, y_pred)
        assert str(metric) == string
