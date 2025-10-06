import numpy as np
import pytest
from conftest import is_activation, is_linear, is_sequential

from dtaianomaly.anomaly_detection import HybridKNearestNeighbors, Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import ACTIVATION_FUNCTIONS
from dtaianomaly.anomaly_detection._HybridKNearestNeighbors import _AutoEncoder
from dtaianomaly.windowing import sliding_window


class TestHybridKNearestNeighbors:

    def test_supervision(self):
        assert HybridKNearestNeighbors(1).supervision == Supervision.SEMI_SUPERVISED

    def test_build_architecture_default(self, univariate_time_series):
        window_size = 32
        detector = HybridKNearestNeighbors(window_size=window_size)
        windows = sliding_window(
            univariate_time_series, window_size=window_size, stride=1
        )
        auto_encoder = detector.build_auto_encoder(windows)

        assert isinstance(auto_encoder, _AutoEncoder)

        encoder_modules = auto_encoder.encoder.modules()
        assert is_sequential(next(encoder_modules))
        assert is_linear(next(encoder_modules), window_size, 64)
        assert is_activation(next(encoder_modules), "relu")
        assert is_linear(next(encoder_modules), 64, 16)
        with pytest.raises(StopIteration):
            next(encoder_modules)

        decoder_modules = auto_encoder.decoder.modules()
        assert is_sequential(next(decoder_modules))
        assert is_linear(next(decoder_modules), 16, 64)
        assert is_activation(next(decoder_modules), "relu")
        assert is_linear(next(decoder_modules), 64, window_size)
        with pytest.raises(StopIteration):
            next(decoder_modules)

    def test_build_architecture_custom_layers(self, univariate_time_series):
        window_size = 32
        detector = HybridKNearestNeighbors(
            window_size=window_size,
            hidden_layer_dimensions=[64, 48, 32, 16],
            latent_space_dimension=8,
        )
        windows = sliding_window(
            univariate_time_series, window_size=window_size, stride=1
        )
        auto_encoder = detector.build_auto_encoder(windows)

        assert isinstance(auto_encoder, _AutoEncoder)

        encoder_modules = auto_encoder.encoder.modules()
        assert is_sequential(next(encoder_modules))
        assert is_linear(next(encoder_modules), window_size, 64)
        assert is_activation(next(encoder_modules), "relu")
        assert is_linear(next(encoder_modules), 64, 48)
        assert is_activation(next(encoder_modules), "relu")
        assert is_linear(next(encoder_modules), 48, 32)
        assert is_activation(next(encoder_modules), "relu")
        assert is_linear(next(encoder_modules), 32, 16)
        assert is_activation(next(encoder_modules), "relu")
        assert is_linear(next(encoder_modules), 16, 8)
        with pytest.raises(StopIteration):
            next(encoder_modules)

        decoder_modules = auto_encoder.decoder.modules()
        assert is_sequential(next(decoder_modules))
        assert is_linear(next(decoder_modules), 8, 16)
        assert is_activation(next(decoder_modules), "relu")
        assert is_linear(next(decoder_modules), 16, 32)
        assert is_activation(next(decoder_modules), "relu")
        assert is_linear(next(decoder_modules), 32, 48)
        assert is_activation(next(decoder_modules), "relu")
        assert is_linear(next(decoder_modules), 48, 64)
        assert is_activation(next(decoder_modules), "relu")
        assert is_linear(next(decoder_modules), 64, window_size)
        with pytest.raises(StopIteration):
            next(decoder_modules)

    @pytest.mark.parametrize("activation_function", ACTIVATION_FUNCTIONS)
    def test_build_architecture_custom_activation_function(
        self, univariate_time_series, activation_function
    ):
        window_size = 32
        detector = HybridKNearestNeighbors(
            window_size=window_size, activation_function=activation_function
        )
        windows = sliding_window(
            univariate_time_series, window_size=window_size, stride=1
        )
        auto_encoder = detector.build_auto_encoder(windows)

        assert isinstance(auto_encoder, _AutoEncoder)

        encoder_modules = auto_encoder.encoder.modules()
        assert is_sequential(next(encoder_modules))
        assert is_linear(next(encoder_modules), window_size, 64)
        assert is_activation(next(encoder_modules), activation_function)
        assert is_linear(next(encoder_modules), 64, 16)
        with pytest.raises(StopIteration):
            next(encoder_modules)

        decoder_modules = auto_encoder.decoder.modules()
        assert is_sequential(next(decoder_modules))
        assert is_linear(next(decoder_modules), 16, 64)
        assert is_activation(next(decoder_modules), activation_function)
        assert is_linear(next(decoder_modules), 64, window_size)
        with pytest.raises(StopIteration):
            next(decoder_modules)

    def test_seed(self, univariate_time_series):
        detector1 = HybridKNearestNeighbors(1, seed=0)
        y_pred1 = detector1.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        detector2 = HybridKNearestNeighbors(1, seed=0)
        y_pred2 = detector2.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        assert np.array_equal(y_pred1, y_pred2)

    @pytest.mark.parametrize(
        "length,max_samples,expected",
        [
            (1000, 50, 50),
            (1000, 250, 250),
            (1000, 0.5, 500),
            (1000, 0.1, 100),
            (1000, "auto", 125),
        ],
    )
    def test_create_subsets(self, length, max_samples, expected):
        detector = HybridKNearestNeighbors(1, max_samples=max_samples, n_estimators=8)
        subsets = detector._create_subsets(np.empty(shape=(length, 1)))
        for subset in subsets:
            assert subset.shape[0] == expected
