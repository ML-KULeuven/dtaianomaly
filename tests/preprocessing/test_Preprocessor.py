import pytest
import numpy as np
import copy

from dtaianomaly.preprocessing import Identity, MinMaxScaler, StandardScaler, MovingAverage, ExponentialMovingAverage, ChainedPreprocessor, NbSamplesUnderSampler, SamplingRateUnderSampler, check_preprocessing_inputs


class TestCheckPreprocessingInputs:

    def test_valid_univariate(self, univariate_time_series):
        y = np.random.default_rng().choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        check_preprocessing_inputs(univariate_time_series, y)

    def test_valid_multivariate(self, multivariate_time_series):
        y = np.random.default_rng().choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        check_preprocessing_inputs(multivariate_time_series, y)

    def test_valid_univariate_list(self):
        check_preprocessing_inputs([1, 2, 3, 4, 5], [0, 1, 0, 1, 0])

    def test_valid_mutlivariate_list(self):
        check_preprocessing_inputs([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], [0, 1, 0, 1, 0])

    def test_invalid_x(self):
        with pytest.raises(ValueError):
            check_preprocessing_inputs(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_x_with_none_y(self):
        with pytest.raises(ValueError):
            check_preprocessing_inputs(
                np.array(['1', '2', '3', '4', '5']),
                None
            )

    def test_invalid_y(self):
        with pytest.raises(ValueError):
            check_preprocessing_inputs(
                np.array([1, 2, 3, 4, 5]),
                np.array(['0', '0', '1', '0', '1'])
            )

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            check_preprocessing_inputs(
                np.array([1, 2, 3, 4, 5]),
                np.array([0, 0, 1, 0, 1, 0])
            )

    def test_invalid_shapes_multivariate(self):
        with pytest.raises(ValueError):
            check_preprocessing_inputs(
                np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]),
                np.array([0, 0, 1, 0, 1, 0])
            )

    def test_none_y(self):
        check_preprocessing_inputs(
            np.array([1, 2, 3, 4, 5]),
            None
        )


@pytest.mark.parametrize('preprocessor', [
    Identity(),
    MinMaxScaler(),
    StandardScaler(),
    MovingAverage(5),
    ExponentialMovingAverage(0.5),
    NbSamplesUnderSampler(100),
    SamplingRateUnderSampler(5),
    ChainedPreprocessor([Identity(), MinMaxScaler()])
])
class TestPreprocessors:

    def test_invalid_input_fit(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.fit(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_input_transform(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.transform(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_invalid_input_fit_transform(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.fit_transform(
                np.array(['1', '2', '3', '4', '5']),
                np.array([0, 0, 1, 0, 1])
            )

    def test_univariate(self, preprocessor, univariate_time_series):
        preprocessor.fit_transform(univariate_time_series)

    def test_multivariate(self, preprocessor, multivariate_time_series):
        preprocessor.fit_transform(multivariate_time_series)

    def test_fit(self, preprocessor, univariate_time_series):
        preprocessor_copy = copy.deepcopy(preprocessor)
        preprocessor.fit(univariate_time_series)
        preprocessor_copy._fit(univariate_time_series)
        if isinstance(preprocessor, ChainedPreprocessor):
            for i in range(len(preprocessor.base_preprocessors)):
                assert preprocessor_copy.base_preprocessors[i].__dict__ == preprocessor.base_preprocessors[i].__dict__
        else:
            assert preprocessor_copy.__dict__ == preprocessor.__dict__

    def test_transform(self, preprocessor, univariate_time_series):
        preprocessor.fit(univariate_time_series)
        preprocessor_copy = copy.deepcopy(preprocessor)
        x, _ = preprocessor.transform(univariate_time_series)
        x_, _ = preprocessor_copy._transform(univariate_time_series)
        assert np.array_equal(x, x_)

    def test_fit_transform_different_time_series(self, preprocessor, univariate_time_series):
        split = int(univariate_time_series.shape[0] / 2)
        x_fit = univariate_time_series[:split]
        x_transform = univariate_time_series[split:]
        preprocessor.fit(x_fit).transform(x_transform)


class TestIdentity:

    def test(self):
        rng = np.random.default_rng()
        x = rng.uniform(size=1000)
        y = rng.choice([0, 1], size=1000, replace=True)
        x_, y_ = Identity().fit_transform(x, y)
        assert np.array_equal(x, x_)
        assert np.array_equal(y, y_)

    def test_str(self):
        assert str(Identity()) == 'Identity()'
