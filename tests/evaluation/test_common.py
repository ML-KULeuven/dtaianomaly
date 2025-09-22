import numpy as np
import pytest

from dtaianomaly.evaluation._common import make_intervals


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
        assert start.shape[0] == end.shape[0] == len(expected)
        for s, e, (s_expected, e_expected) in zip(start, end, expected):
            assert s == s_expected
            assert e == e_expected
