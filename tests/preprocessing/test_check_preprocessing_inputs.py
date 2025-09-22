import pytest
import numpy as np

from dtaianomaly.preprocessing._Preprocessor import _check_preprocessing_inputs


class TestCheckPreprocessingInputs:

    def test_valid_univariate(self, univariate_time_series):
        y = np.random.default_rng().choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        _check_preprocessing_inputs(univariate_time_series, y)

    def test_valid_multivariate(self, multivariate_time_series):
        y = np.random.default_rng().choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        _check_preprocessing_inputs(multivariate_time_series, y)

    def test_valid_univariate_list(self):
        _check_preprocessing_inputs([1, 2, 3, 4, 5], [0, 1, 0, 1, 0])

    def test_valid_mutlivariate_list(self):
        _check_preprocessing_inputs([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], [0, 1, 0, 1, 0])

    def test_invalid_x(self):
        with pytest.raises(ValueError):
            _check_preprocessing_inputs(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_x_with_none_y(self):
        with pytest.raises(ValueError):
            _check_preprocessing_inputs(
                np.array(['1', '2', '3', '4', '5']),
                None
            )

    def test_invalid_y(self):
        with pytest.raises(ValueError):
            _check_preprocessing_inputs(
                np.array([1, 2, 3, 4, 5]),
                np.array(['0', '0', '1', '0', '1'])
            )

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            _check_preprocessing_inputs(
                np.array([1, 2, 3, 4, 5]),
                np.array([0, 0, 1, 0, 1, 0])
            )

    def test_invalid_shapes_multivariate(self):
        with pytest.raises(ValueError):
            _check_preprocessing_inputs(
                np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]),
                np.array([0, 0, 1, 0, 1, 0])
            )

    def test_none_y(self):
        _check_preprocessing_inputs(
            np.array([1, 2, 3, 4, 5]),
            None
        )
