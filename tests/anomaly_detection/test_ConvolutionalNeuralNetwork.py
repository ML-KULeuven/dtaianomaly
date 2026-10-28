import pytest
from conftest import (
    is_activation,
    is_avg_pooling,
    is_batch_normalization,
    is_conv1d,
    is_flatten,
    is_linear,
    is_sequential,
)

from dtaianomaly.anomaly_detection import ConvolutionalNeuralNetwork
from dtaianomaly.anomaly_detection._BaseNeuralDetector import ACTIVATION_FUNCTIONS
from dtaianomaly.anomaly_detection._ConvolutionalNeuralNetwork import _CNN


class TestInitialize:

    def test_kernel_size_valid(self):
        detector = ConvolutionalNeuralNetwork(window_size=16, kernel_size=8)
        assert detector.kernel_size == 8

    def test_kernel_size_invalid_type(self):
        with pytest.raises(TypeError):
            ConvolutionalNeuralNetwork(window_size=16, kernel_size=32.0)

    def test_kernel_size_invalid_value(self):
        with pytest.raises(ValueError):
            ConvolutionalNeuralNetwork(window_size=16, kernel_size=0)

    def test_hidden_layers_valid(self):
        detector = ConvolutionalNeuralNetwork(window_size=16, hidden_layers=[32])
        assert detector.hidden_layers == [32]

    def test_hidden_layers_invalid_type(self):
        with pytest.raises(TypeError):
            ConvolutionalNeuralNetwork(window_size=16, hidden_layers=["32", 16, 8])

    @pytest.mark.parametrize("dimensions", [[32, -16, 8], []])
    def test_hidden_layers_invalid_value(self, dimensions):
        with pytest.raises(ValueError):
            ConvolutionalNeuralNetwork(window_size=16, hidden_layers=dimensions)


class TestBuildArchitecture:

    def test_default(self):
        detector = ConvolutionalNeuralNetwork(window_size=16)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _CNN)

        assert is_sequential(next(modules))

        assert is_conv1d(next(modules), 8, 64, (3,), (1,))
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        assert is_conv1d(next(modules), 64, 32, (3,), (1,))
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        assert is_flatten(next(modules))
        assert is_linear(next(modules), 32 * 4, 8)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("kernel_size", [5, 9, 12])
    def test_custom_kernel_size(self, kernel_size):
        detector = ConvolutionalNeuralNetwork(window_size=16, kernel_size=kernel_size)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _CNN)

        assert is_sequential(next(modules))

        assert is_conv1d(
            next(modules), 8, 64, (kernel_size,), (int((kernel_size - 1) // 2),)
        )
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        assert is_conv1d(
            next(modules), 64, 32, (kernel_size,), (int((kernel_size - 1) // 2),)
        )
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        assert is_flatten(next(modules))
        assert is_linear(next(modules), 32 * 4, 8)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("hidden_layers", [[64], [64, 32, 16]])
    def test_custom_hidden_layers(self, hidden_layers):
        detector = ConvolutionalNeuralNetwork(
            window_size=16, hidden_layers=hidden_layers
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _CNN)

        assert is_sequential(next(modules))

        assert is_conv1d(next(modules), 8, hidden_layers[0], (3,), (1,))
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        for d, d_ in zip(hidden_layers[:-1], hidden_layers[1:]):
            assert is_conv1d(next(modules), d, d_, (3,), (1,))
            assert is_batch_normalization(next(modules), d_)
            assert is_activation(next(modules), "relu")
            assert is_avg_pooling(next(modules))

        assert is_flatten(next(modules))
        assert is_linear(
            next(modules), hidden_layers[-1] * int(16 / 2 ** len(hidden_layers)), 8
        )

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("activation_function", ACTIVATION_FUNCTIONS)
    def test_custom_activation_function(self, activation_function):
        detector = ConvolutionalNeuralNetwork(
            window_size=16, activation_function=activation_function
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _CNN)

        assert is_sequential(next(modules))

        assert is_conv1d(next(modules), 8, 64, (3,), (1,))
        assert is_activation(next(modules), activation_function)
        assert is_avg_pooling(next(modules))

        assert is_conv1d(next(modules), 64, 32, (3,), (1,))
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), activation_function)
        assert is_avg_pooling(next(modules))

        assert is_flatten(next(modules))
        assert is_linear(next(modules), 32 * 4, 8)

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_batch_normalization(self):
        detector = ConvolutionalNeuralNetwork(window_size=16, batch_normalization=False)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _CNN)

        assert is_sequential(next(modules))

        assert is_conv1d(next(modules), 8, 64, (3,), (1,))
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        assert is_conv1d(next(modules), 64, 32, (3,), (1,))
        # No batch norm here
        assert is_activation(next(modules), "relu")
        assert is_avg_pooling(next(modules))

        assert is_flatten(next(modules))
        assert is_linear(next(modules), 32 * 4, 8)

        with pytest.raises(StopIteration):
            next(modules)
