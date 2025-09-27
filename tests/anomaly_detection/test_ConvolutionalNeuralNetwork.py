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

from dtaianomaly.anomaly_detection import ConvolutionalNeuralNetwork, Supervision
from dtaianomaly.anomaly_detection._BaseNeuralDetector import ACTIVATION_FUNCTIONS
from dtaianomaly.anomaly_detection._ConvolutionalNeuralNetwork import _CNN


class TestInitialize:

    @pytest.mark.parametrize("kernel_size", [4, 8, 16, 32])
    def test_kernel_size_valid(self, kernel_size):
        detector = ConvolutionalNeuralNetwork(window_size=16, kernel_size=kernel_size)
        assert detector.kernel_size == kernel_size

    @pytest.mark.parametrize("kernel_size", [[32], "32", 32.0, True])
    def test_kernel_size_invalid_type(self, kernel_size):
        with pytest.raises(TypeError):
            ConvolutionalNeuralNetwork(window_size=16, kernel_size=kernel_size)

    @pytest.mark.parametrize("kernel_size", [0, -1])
    def test_kernel_size_invalid_value(self, kernel_size):
        with pytest.raises(ValueError):
            ConvolutionalNeuralNetwork(window_size=16, kernel_size=kernel_size)

    @pytest.mark.parametrize("dimensions", [[32], [32, 16], [32, 16, 8], [8, 16, 32]])
    def test_hidden_layers_valid(self, dimensions):
        detector = ConvolutionalNeuralNetwork(window_size=16, hidden_layers=dimensions)
        assert detector.hidden_layers == dimensions

    @pytest.mark.parametrize("dimensions", [32, ["32", 16, 8]])
    def test_hidden_layers_invalid_type(self, dimensions):
        with pytest.raises(TypeError):
            ConvolutionalNeuralNetwork(window_size=16, hidden_layers=dimensions)

    @pytest.mark.parametrize("dimensions", [[32, -16, 8], []])
    def test_hidden_layers_invalid_value(self, dimensions):
        with pytest.raises(ValueError):
            ConvolutionalNeuralNetwork(window_size=16, hidden_layers=dimensions)

    @pytest.mark.parametrize("activation_function", ACTIVATION_FUNCTIONS)
    def test_activation_function_valid(self, activation_function):
        detector = ConvolutionalNeuralNetwork(
            window_size=16, activation_function=activation_function
        )
        assert detector.activation_function == activation_function

    @pytest.mark.parametrize("activation_function", [1, ["tanh", 1, "sigmoid"]])
    def test_activation_functions_invalid_type(self, activation_function):
        with pytest.raises(TypeError):
            ConvolutionalNeuralNetwork(
                window_size=16, activation_function=activation_function
            )

    @pytest.mark.parametrize("activation_function", ["invalid-value"])
    def test_activation_functions_invalid_value(self, activation_function):
        with pytest.raises(ValueError):
            ConvolutionalNeuralNetwork(
                window_size=16, activation_function=activation_function
            )

    @pytest.mark.parametrize("batch_normalization", [True, False])
    def test_batch_normalization_valid(self, batch_normalization):
        detector = ConvolutionalNeuralNetwork(
            window_size=16, batch_normalization=batch_normalization
        )
        assert detector.batch_normalization == batch_normalization

    @pytest.mark.parametrize("batch_normalization", ["True", 1, 1.0])
    def test_batch_normalization_invalid_type(self, batch_normalization):
        with pytest.raises(TypeError):
            ConvolutionalNeuralNetwork(
                window_size=16, batch_normalization=batch_normalization
            )


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
