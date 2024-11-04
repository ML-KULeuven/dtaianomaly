
import numpy as np
from typing import Any

from dtaianomaly.utils import is_valid_list, is_valid_array_like, is_univariate


class TestIsValidList:

    def test_empty_list(self):
        assert is_valid_list([], Any)

    def test_np_array(self):
        assert not is_valid_list(np.array([1, 2, 3, 4, 5, 6]), Any)

    def test_valid(self):
        assert is_valid_list([1, 2, 3, 4, 5, 6], int)

    def test_invalid(self):
        assert not is_valid_list([1, 2, 3, '4', 5, 6], int)


class TestIsValidArrayLike:

    def test_empty_list(self):
        assert is_valid_array_like([])

    def test_empty_np_array(self):
        assert is_valid_array_like(np.array([]))

    def test_valid_list_int(self):
        assert is_valid_array_like([1, 2, 3, 4, 5])

    def test_valid_list_int_2d(self):
        assert is_valid_array_like([[1], [2], [3], [4], [5]])

    def test_valid_np_array_int(self):
        assert is_valid_array_like(np.array([1, 2, 3, 4, 5]))

    def test_valid_np_array_int_2d(self):
        assert is_valid_array_like(np.array([[1], [2], [3], [4], [5]]))

    def test_valid_list_float(self):
        assert is_valid_array_like([1.9, 2.8, 3.7, 4.6, 5.5])

    def test_valid_np_array_float(self):
        assert is_valid_array_like(np.array([1.9, 2.8, 3.7, 4.6, 5.5]))

    def test_valid_list_bool(self):
        assert is_valid_array_like([True, True, False, True, False])

    def test_valid_np_array_bool(self):
        assert is_valid_array_like(np.array([True, True, False, True, False]))

    def test_valid_list_mixed_type(self):
        assert is_valid_array_like([1.9, 2, True, 4, 5.5])

    def test_valid_np_array_mixed_type(self):
        assert is_valid_array_like(np.array([1.9, 2, True, 4, 5.5]))

    def test_valid_multivariate_list_int(self):
        assert is_valid_array_like([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

    def test_valid_multivariate_np_array_int(self):
        assert is_valid_array_like(np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]))

    def test_valid_multivariate_list_float(self):
        assert is_valid_array_like([[1.9, 9.1], [2.8, 8.2], [3.7, 7.3], [4.6, 6.4], [5.5, 5.5]])

    def test_valid_multivariate_np_array_float(self):
        assert is_valid_array_like(np.array([[1.9, 9.1], [2.8, 8.2], [3.7, 7.3], [4.6, 6.4], [5.5, 5.5]]))

    def test_valid_multivariate_list_bool(self):
        assert is_valid_array_like([[True, True], [True, False], [False, False], [True, False], [False, True]])

    def test_valid_multivariate_np_array_bool(self):
        assert is_valid_array_like(np.array([[True, True], [True, False], [False, False], [True, False], [False, True]]))

    def test_valid_multivariate_list_mixed_type(self):
        assert is_valid_array_like([[1.9, 1], [2, False], [True, 3], [4, 6.4], [5.5, True]])

    def test_valid_multivariate_np_array_mixed_type(self):
        assert is_valid_array_like(np.array([[1.9, 1], [2, False], [True, 3], [4, 6.4], [5.5, True]]))

    def test_invalid_list(self):
        assert not is_valid_array_like([1, 2, 3, 4, '5'])

    def test_invalid_np_array(self):
        assert not is_valid_array_like(np.array([1, 2, 3, 4, '5']))

    def test_invalid_int(self):
        assert not is_valid_array_like(1)

    def test_invalid_float(self):
        assert not is_valid_array_like(1.9)

    def test_invalid_str(self):
        assert not is_valid_array_like('1')

    def test_invalid_bool(self):
        assert not is_valid_array_like(True)

    def test_invalid_multivariate_list_type(self):
        assert not is_valid_array_like([[1, 10], [2, 20], [3, 30], [4, 40], [5, '50']])

    def test_invalid_multivariate_with_str(self):
        assert not is_valid_array_like([[1, 10], [2, 20], [3, 30], [4, 40], '55'])

    def test_invalid_multivariate_list_dimension(self):
        assert not is_valid_array_like([[1, 10], [2, 20], [3, 30], [4, 40, 400], [5, 50]])

    def test_invalid_multivariate_list_non_list(self):
        assert not is_valid_array_like([[1], [2], 3, [4], [5]])

    def test_invalid_multivariate_list_first_non_list(self):
        assert not is_valid_array_like([1, [2], [3], [4], [5]])


class TestIsUnivariate:

    def test_multivariate(self, multivariate_time_series):
        assert not is_univariate(multivariate_time_series)

    def test_univariate_1_dimension(self, univariate_time_series):
        univariate_time_series = univariate_time_series.squeeze()
        assert len(univariate_time_series.shape) == 1
        assert is_univariate(univariate_time_series)

    def test_univariate_2_dimensions(self, univariate_time_series):
        univariate_time_series = univariate_time_series.reshape(univariate_time_series.shape[0], 1)
        assert len(univariate_time_series.shape) == 2
        assert is_univariate(univariate_time_series)

    def test_multivariate_list(self):
        assert not is_univariate([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

    def test_univariate_list(self, univariate_time_series):
        assert is_univariate([i for i in univariate_time_series])
