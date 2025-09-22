from typing import Any

import numpy as np
import pytest

from dtaianomaly.utils import (
    get_dimension,
    is_univariate,
    is_valid_array_like,
    is_valid_list,
)


class TestIsValidList:

    @pytest.mark.parametrize(
        "value,target_type",
        [
            ([], Any),
            ([1, 2, 3, 4, 5, 6], int),
            ((1, 3, 6, 7), int),
            (("some-str",), str),
        ],
    )
    def test_valid(self, value, target_type):
        assert is_valid_list(value, target_type)

    @pytest.mark.parametrize(
        "value,target_type",
        [
            (np.array([1, 2, 3, 4, 5, 6]), Any),
            ([1, 2, 3, "4", 5, 6], int),
        ],
    )
    def test_invalid(self, value, target_type):
        assert not is_valid_list([1, 2, 3, "4", 5, 6], int)


class TestIsValidArrayLike:

    @pytest.mark.parametrize(
        "value",
        [
            [],
            [1, 2, 3, 4, 5],
            [[1], [2], [3], [4], [5]],
            [1.9, 2.8, 3.7, 4.6, 5.5],
            [True, True, False, True, False],
            [1.9, 2, True, 4, 5.5],
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
            [[1.9, 9.1], [2.8, 8.2], [3.7, 7.3], [4.6, 6.4], [5.5, 5.5]],
            [[True, True], [True, False], [False, False], [True, False], [False, True]],
            [[1.9, 1], [2, False], [True, 3], [4, 6.4], [5.5, True]],
            [
                np.datetime64("2005-02-25"),
                np.datetime64("2005-02-26"),
                np.datetime64("2005-02-27"),
            ],
            [
                [np.datetime64("2005-02-25"), np.datetime64("2005-02-25")],
                [np.datetime64("2005-02-26"), np.datetime64("2005-02-26")],
                [np.datetime64("2005-02-27"), np.datetime64("2005-02-27")],
            ],
        ],
    )
    def test_valid(self, value):
        assert is_valid_array_like(value)
        assert is_valid_array_like(np.array(value))

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3, 4, "5"],
            np.array([1, 2, 3, 4, "5"]),
            1,
            1.9,
            "1",
            True,
            False,
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, "50"]],
            [[1, 10], [2, 20], [3, 30], [4, 40], "55"],
            [[1, 10], [2, 20], [3, 30], [4, 40, 400], [5, 50]],
            [[1], [2], 3, [4], [5]],
            [1, [2], [3], [4], [5]],
        ],
    )
    def test_invalid(self, value):
        assert not is_valid_array_like(value)


class TestIsUnivariate:

    @pytest.mark.parametrize(
        "X",
        [
            np.arange(100).squeeze(),
            np.arange(100).reshape(100, 1),
            list(np.arange(100)),
        ],
    )
    def test_univariate(self, X):
        assert is_univariate(X)

    @pytest.mark.parametrize(
        "X",
        [
            np.arange(100).reshape(50, 2),
            [[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
        ],
    )
    def test_multivariate(self, X):
        assert not is_univariate(X)


class TestGetDimension:

    @pytest.mark.parametrize("dimension", [1, 2, 3, 5, 10])
    def test(self, dimension):
        X = np.random.default_rng(42).uniform(size=(1000, dimension))
        assert get_dimension(X) == dimension
