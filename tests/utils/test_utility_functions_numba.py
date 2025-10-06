import numpy as np
import pytest

from dtaianomaly.utils import make_intervals, np_any_axis0, np_any_axis1, np_diff

_SEEDS = list(range(5))


@pytest.mark.numba
@pytest.mark.parametrize("seed", _SEEDS)
class TestGenericFunctions:

    def test_np_any_axis1(self, seed):
        array = np.random.default_rng(seed).choice([True, False], size=(100, 3))
        assert np.array_equal(np.any(array, axis=1), np_any_axis1(array))

    def test_np_any_axis0(self, seed):
        array = np.random.default_rng(seed).choice([True, False], size=(3, 100))
        assert np.array_equal(np.any(array, axis=0), np_any_axis0(array))

    def test_np_diff(self, seed):
        array = np.random.default_rng(seed).uniform(low=-1.0, high=1.0, size=100)
        assert np.array_equal(np.diff(array), np_diff(array))


@pytest.mark.numba
class TestMakeIntervals:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ([], []),
            ([0, 0, 0], []),
            ([1, 1, 1], [(0, 2)]),
            ([0, 1, 1, 0], [(1, 2)]),
            ([1, 1, 0, 1], [(0, 1), (3, 3)]),
            ([0, 1, 0, 1, 1, 0], [(1, 1), (3, 4)]),
            ([1, 0, 0, 1, 1, 1], [(0, 0), (3, 5)]),
            ([1], [(0, 0)]),
            ([0], []),
            ([0, 1, 1, 1, 0, 0, 1, 0], [(1, 3), (6, 6)]),
        ],
    )
    def test_make_intervals(self, value, expected):
        start, end = make_intervals(np.array(value))
        assert start.shape[0] == len(expected)
        assert end.shape[0] == len(expected)
        for s, e, (s_expected, e_expected) in zip(start, end, expected):
            assert s == s_expected
            assert e == e_expected
