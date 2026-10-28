import pytest
from conftest import (
    is_activation,
    is_batch_normalization,
    is_dropout,
    is_flatten,
    is_linear,
    is_sequential,
    is_un_flatten,
)

from dtaianomaly.anomaly_detection import MultilayerPerceptron
from dtaianomaly.anomaly_detection._BaseNeuralDetector import ACTIVATION_FUNCTIONS


class TestInitialize:

    def test_hidden_layers_valid(self):
        detector = MultilayerPerceptron(window_size=16, hidden_layers=[32, 16, 8])
        assert detector.hidden_layers == [32, 16, 8]

    def test_hidden_layers_empty_valid(self):
        detector = MultilayerPerceptron(window_size=16, hidden_layers=[])
        assert detector.hidden_layers == []

    def test_hidden_layers_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, hidden_layers=["32", 16, 8])

    def test_hidden_layers_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, hidden_layers=[32, -16, 8])


class TestBuildArchitecture:

    def test_default(self):
        detector = MultilayerPerceptron(window_size=16)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_activation(next(modules), "relu")

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("forecast_length", [1, 2, 4, 8])
    def test_custom_forecast_length(self, forecast_length):
        detector = MultilayerPerceptron(
            window_size=16,
            forecast_length=forecast_length,
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8 * forecast_length)
        assert is_activation(next(modules), "relu")

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("hidden_layers", [[], [64], [32, 64], [16, 32, 64]])
    def test_custom_hidden_layers(self, hidden_layers):
        detector = MultilayerPerceptron(
            window_size=16,
            hidden_layers=hidden_layers,
            batch_normalization=False,  # For simpler testing
            dropout_rate=0.0,  # For simpler testing
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        dimensions = [128, *hidden_layers, 8]
        for d, d_ in zip(dimensions[:-1], dimensions[1:]):
            assert is_linear(next(modules), d, d_)
            assert is_activation(next(modules), "relu")

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_batch_normalization(self):
        detector = MultilayerPerceptron(window_size=16, batch_normalization=False)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_activation(next(modules), "relu")

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("activation_function", ACTIVATION_FUNCTIONS)
    def test_custom_activation_function(self, activation_function):
        detector = MultilayerPerceptron(
            window_size=16, activation_function=activation_function
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), activation_function)
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), activation_function)
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_activation(next(modules), activation_function)

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("dropout_rate", [0.1, 0.3])
    def test_custom_dropout_rate(self, dropout_rate):
        detector = MultilayerPerceptron(window_size=16, dropout_rate=dropout_rate)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), dropout_rate)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), "relu")
        assert is_dropout(next(modules), dropout_rate)

        assert is_linear(next(modules), 32, 8)
        assert is_activation(next(modules), "relu")

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    def test_zero_dropout_rate(self):
        detector = MultilayerPerceptron(window_size=16, dropout_rate=0.0)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), "relu")

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), "relu")

        assert is_linear(next(modules), 32, 8)
        assert is_activation(next(modules), "relu")

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)
