import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from dtaianomaly.anomaly_detection import HydraAD, Supervision
from dtaianomaly.anomaly_detection._HydraAD import _sparse_scaling, _standard_scaling
from dtaianomaly.windowing import sliding_window


def _sine(n=300, period=20):
    return np.sin(2 * np.pi * np.arange(n) / period)


def _multivariate_sines(n=600, periods=(20, 50, 30)):
    return np.stack([_sine(n, period) for period in periods], axis=1)


class TestHydraAD:

    def test_supervision(self):
        assert HydraAD(16).supervision == Supervision.UNSUPERVISED

    def test_seed(self, univariate_time_series):
        detector1 = HydraAD(64, n_groups=4, seed=0)
        y_pred1 = detector1.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        detector2 = HydraAD(64, n_groups=4, seed=0)
        y_pred2 = detector2.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        assert np.array_equal(y_pred1, y_pred2)

    def test_different_seeds(self, univariate_time_series):
        detector1 = HydraAD(64, n_groups=4, seed=0)
        y_pred1 = detector1.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        detector2 = HydraAD(64, n_groups=4, seed=1)
        y_pred2 = detector2.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        assert not np.array_equal(y_pred1, y_pred2)

    def test_seed_does_not_change_global_torch_state(self):
        state = torch.get_rng_state()
        HydraAD(16, n_groups=4, seed=0).fit(_sine())
        assert torch.equal(torch.get_rng_state(), state)

    @pytest.mark.parametrize("stride", [1, 3, 16])
    def test_integer_stride(self, stride):
        detector = HydraAD(16, stride=stride, n_groups=2).fit(_sine())
        assert detector.stride_ == stride

    @pytest.mark.parametrize("stride,expected", [(0.25, 8), (0.5, 16), (1.0, 32)])
    def test_float_stride(self, stride, expected):
        detector = HydraAD(32, stride=stride, n_groups=2).fit(_sine())
        assert detector.stride_ == expected

    def test_float_stride_at_least_one(self):
        detector = HydraAD(16, stride=0.01, n_groups=2).fit(_sine())
        assert detector.stride_ == 1

    @pytest.mark.parametrize("stride", [0, 1.5, -1, -0.5])
    def test_invalid_stride(self, stride):
        with pytest.raises(ValueError, match="Attribute 'stride'"):
            HydraAD(16, stride=stride)

    @pytest.mark.parametrize(
        "attribute,value,error",
        [
            ("quantile", -0.1, ValueError),
            ("quantile", 1.1, ValueError),
            ("scaling", "foo", ValueError),
            ("kernel_size", 1, ValueError),
            ("kernel_size", 0, ValueError),
            ("dilations", [0], ValueError),
            ("dilations", [1, 0], ValueError),
            ("dilations", "foo", ValueError),
            ("periodicity", 0.0, ValueError),
            ("periodicity", -1.0, ValueError),
            ("relative_n_channels", 0.0, ValueError),
            ("relative_n_channels", -0.1, ValueError),
            ("relative_n_channels", 1.1, ValueError),
            ("max_n_channels", 0, ValueError),
        ],
    )
    def test_invalid_attribute(self, attribute, value, error):
        with pytest.raises(error, match=f"Attribute '{attribute}'"):
            HydraAD(16, **{attribute: value})

    def test_empty_dilations_invalid(self):
        with pytest.raises(ValueError, match="Attribute 'dilations'"):
            HydraAD(16, dilations=[])

    @pytest.mark.parametrize(
        "device,error,match",
        [
            ("foo", ValueError, "Invalid input device"),
            (3, TypeError, "should be a string"),
        ],
    )
    def test_invalid_device(self, device, error, match):
        with pytest.raises(error, match=match):
            HydraAD(16, device=device)

    def test_window_size_at_least_kernel_size(self):
        detector = HydraAD(4, kernel_size=9, n_groups=2).fit(_sine())
        assert detector.window_size_ == 9

    @pytest.mark.parametrize("periodicity", [0.5, 1.0, 2.0, 3.0])
    def test_periodicity_multiplies_window_size(self, periodicity):
        detector = HydraAD("fft", periodicity=periodicity, n_groups=2).fit(_sine())
        assert detector.window_size_ == int(20 * periodicity)

    @pytest.mark.parametrize("window_size", [9, 16, 17, 33, 64, 100])
    def test_auto_dilations(self, window_size):
        detector = HydraAD(window_size, n_groups=2).fit(_sine())
        dilations = detector.hydra_.dilations_.tolist()
        assert dilations == [2**i for i in range(len(dilations))]
        assert (detector.kernel_size - 1) * dilations[-1] + 1 <= window_size
        assert (detector.kernel_size - 1) * 2 * dilations[-1] + 1 > window_size

    def test_explicit_dilations(self):
        detector = HydraAD(32, dilations=[3, 1, 5], n_groups=2).fit(_sine())
        assert detector.hydra_.dilations_.tolist() == [3, 1, 5]

    @pytest.mark.parametrize("scaling", ["sparse", "standard", "none"])
    def test_scaling_univariate(self, scaling, univariate_time_series):
        detector = HydraAD(64, scaling=scaling, n_groups=4, seed=0)
        scores = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert scores.shape == (univariate_time_series.shape[0],)
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize("scaling", ["sparse", "standard", "none"])
    def test_scaling_multivariate(self, scaling, multivariate_time_series):
        detector = HydraAD(32, scaling=scaling, n_groups=4, seed=0)
        scores = detector.fit(multivariate_time_series).decision_function(
            multivariate_time_series
        )
        assert scores.shape == (multivariate_time_series.shape[0],)
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize("n", [300, 301, 307])
    @pytest.mark.parametrize("stride", [1, 3, 7, 16])
    def test_no_nan_for_unaligned_stride(self, n, stride):
        X = _sine(n)
        scores = HydraAD(16, stride=stride, n_groups=2, seed=0).fit(X)
        assert not np.isnan(scores.decision_function(X)).any()

    @pytest.mark.parametrize(
        "n,window_size,stride",
        [(100, 10, 1), (100, 10, 3), (100, 10, 10), (101, 10, 5), (10, 10, 2)],
    )
    def test_aggregate_windows_matches_sliding_window(self, n, window_size, stride):
        counts = np.random.default_rng(0).standard_normal((n, 3))
        detector = HydraAD(window_size)
        windows = detector._aggregate_windows(
            torch.as_tensor(counts), window_size, stride
        )
        expected = sliding_window(counts, window_size, stride)
        expected = expected.reshape(-1, window_size, 3).sum(axis=1)
        np.testing.assert_allclose(windows.numpy(), expected)

    def test_multivariate_window_size_is_mean(self):
        detector = HydraAD("fft", n_groups=2).fit(_multivariate_sines())
        assert detector.window_size_ == int(np.mean([20, 50, 30]))

    def test_multivariate_window_size_default(self):
        detector = HydraAD("fft", n_groups=2)
        detector.fit(np.zeros((300, 3)), default_window_size=20)
        assert detector.window_size_ == 20

    def test_multivariate_window_size_no_default(self):
        with pytest.raises(ValueError, match="Could not compute the window size"):
            HydraAD("fft", n_groups=2).fit(np.zeros((300, 3)))

    def test_no_first_order_difference(self):
        detector = HydraAD(16, n_groups=5, first_order_difference=False).fit(_sine())
        assert not detector.hydra_.group_use_diff_.any()

    @pytest.mark.parametrize("n_groups", [2, 4, 5])
    def test_first_order_difference(self, n_groups):
        detector = HydraAD(16, n_groups=n_groups, first_order_difference=True)
        detector.fit(_sine())
        expected = [i >= n_groups - n_groups // 2 for i in range(n_groups)]
        assert detector.hydra_.group_use_diff_.tolist() == expected

    def test_first_order_difference_single_group(self):
        detector = HydraAD(16, n_groups=1, first_order_difference=True).fit(_sine())
        assert not detector.hydra_.group_use_diff_.any()


class TestStandardScaling:

    def test_matches_sklearn(self):
        X = np.random.default_rng(0).standard_normal((50, 4)) * 3 + 2
        X[:, 1] = 5.0
        X[:, 2] = 0.0
        scaled = _standard_scaling(torch.as_tensor(X))
        np.testing.assert_allclose(
            scaled.numpy(), StandardScaler().fit_transform(X), atol=1e-10
        )


class TestSparseScaling:

    def test_zeros_remain_zero(self):
        X = np.abs(np.random.default_rng(0).standard_normal((50, 4)))
        X[::3, 0] = 0.0
        X[:, 1] = 0.0
        X[1::2, 2] = 0.0
        scaled = _sparse_scaling(torch.as_tensor(X))
        assert (scaled.numpy()[X == 0] == 0).all()

    def test_finite(self):
        X = np.abs(np.random.default_rng(0).standard_normal((50, 4)))
        X[:, 1] = 0.0
        assert torch.isfinite(_sparse_scaling(torch.as_tensor(X))).all()
