import numpy as np
import pytest

from dtaianomaly.anomaly_detection import DWT_MLEAD, Supervision
from dtaianomaly.anomaly_detection._DWT_MLEAD import _multilevel_haar_transform


@pytest.mark.numba
class TestDWTMLEAD:

    def test_supervision(self):
        detector = DWT_MLEAD()
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_initialize_non_int_start_level(self):
        with pytest.raises(TypeError):
            DWT_MLEAD(start_level=True)
        with pytest.raises(TypeError):
            DWT_MLEAD(start_level="a string")
        DWT_MLEAD(start_level=5)  # Doesn't raise an error

    def test_initialize_too_small_start_level(self):
        with pytest.raises(ValueError):
            DWT_MLEAD(start_level=-1)
        DWT_MLEAD(start_level=5)  # Doesn't raise an error
        DWT_MLEAD(start_level=0)  # Doesn't raise an error

    def test_valid_start_level(self):
        DWT_MLEAD(start_level=1)
        DWT_MLEAD(start_level=10)
        DWT_MLEAD(start_level=100)

    def test_initialize_non_str_quantile_boundary_type(self):
        with pytest.raises(TypeError):
            DWT_MLEAD(quantile_boundary_type=5)
        with pytest.raises(TypeError):
            DWT_MLEAD(quantile_boundary_type=True)
        with pytest.raises(TypeError):
            DWT_MLEAD(quantile_boundary_type=3.14)
        DWT_MLEAD(quantile_boundary_type="percentile")

    def test_initialize_invalid_quantile_boundary_type(self):
        with pytest.raises(ValueError):
            DWT_MLEAD(quantile_boundary_type="something-invalid")
        with pytest.raises(ValueError):
            DWT_MLEAD(quantile_boundary_type="tmp")
        DWT_MLEAD(quantile_boundary_type="percentile")

    def test_initialize_not_implemented_quantile_boundary_type(self):
        with pytest.raises(ValueError):
            DWT_MLEAD(quantile_boundary_type="monte-carlo")
        DWT_MLEAD(quantile_boundary_type="percentile")

    def test_initialize_non_numeric_quantile_epsilon(self):
        with pytest.raises(TypeError):
            DWT_MLEAD(quantile_epsilon="5")
        with pytest.raises(TypeError):
            DWT_MLEAD(quantile_epsilon=True)
        DWT_MLEAD(quantile_epsilon=0.5)
        DWT_MLEAD(quantile_epsilon=0.0)

    def test_initialize_invalid_quantile_epsilon(self):
        with pytest.raises(ValueError):
            DWT_MLEAD(quantile_epsilon=-0.5)
        with pytest.raises(ValueError):
            DWT_MLEAD(quantile_epsilon=1.1)
        DWT_MLEAD(quantile_epsilon=1.0)
        DWT_MLEAD(quantile_epsilon=0.0)

    def test_initialize_non_str_padding_mode(self):
        with pytest.raises(TypeError):
            DWT_MLEAD(padding_mode=5)
        with pytest.raises(TypeError):
            DWT_MLEAD(padding_mode=True)
        with pytest.raises(TypeError):
            DWT_MLEAD(padding_mode=3.14)
        DWT_MLEAD(padding_mode="wrap")

    def test_initialize_invalid_padding_mode(self):
        with pytest.raises(ValueError):
            DWT_MLEAD(padding_mode="something-invalid")
        with pytest.raises(ValueError):
            DWT_MLEAD(padding_mode="tmp")
        DWT_MLEAD(padding_mode="wrap")

    @pytest.mark.parametrize(
        "mode",
        [
            "constant",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
            "empty",
        ],
    )
    def test_initialize_valid_padding_mode(self, mode):
        DWT_MLEAD(padding_mode=mode)

    def test_too_large_start_window(self):
        with pytest.raises(ValueError):
            DWT_MLEAD(start_level=6).decision_function(np.zeros(shape=64))
        DWT_MLEAD(start_level=6).decision_function(np.zeros(shape=65))

        DWT_MLEAD(start_level=5).decision_function(np.zeros(shape=64))
        DWT_MLEAD(start_level=5).decision_function(np.zeros(shape=33))
        with pytest.raises(ValueError):
            DWT_MLEAD(start_level=5).decision_function(np.zeros(shape=32))

    def test_multilevel_haar_transform_invalid_max_levels(self):
        with pytest.raises(ValueError):
            _multilevel_haar_transform(np.zeros(shape=63), levels=6)
        _multilevel_haar_transform(np.zeros(shape=64), levels=6)
