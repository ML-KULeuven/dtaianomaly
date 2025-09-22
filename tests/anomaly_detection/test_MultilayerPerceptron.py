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

from dtaianomaly.anomaly_detection import (
    BaseNeuralDetector,
    MultilayerPerceptron,
    Supervision,
)


class TestMultilayerPerceptron:

    def test_supervision(self):
        assert (
            MultilayerPerceptron(window_size=100).supervision
            == Supervision.SEMI_SUPERVISED
        )

    def test_str(self):
        assert (
            str(MultilayerPerceptron(window_size=100))
            == "MultilayerPerceptron(window_size=100)"
        )
        assert (
            str(MultilayerPerceptron(window_size=100, hidden_layers=[64, 32, 16]))
            == "MultilayerPerceptron(window_size=100,hidden_layers=[64, 32, 16])"
        )
        assert (
            str(MultilayerPerceptron(window_size=100, dropout_rate=0.5))
            == "MultilayerPerceptron(window_size=100,dropout_rate=0.5)"
        )


class TestInitialize:

    @pytest.mark.parametrize("dimensions", [[32], [32, 16], [32, 16, 8], [8, 16, 32]])
    def test_hidden_layers_valid(self, dimensions):
        detector = MultilayerPerceptron(window_size=16, hidden_layers=dimensions)
        assert detector.hidden_layers == dimensions

    @pytest.mark.parametrize("dimensions", [32, ["32", 16, 8]])
    def test_hidden_layers_invalid_type(self, dimensions):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, hidden_layers=dimensions)

    @pytest.mark.parametrize("dimensions", [[32, -16, 8]])
    def test_hidden_layers_invalid_value(self, dimensions):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, hidden_layers=dimensions)

    @pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.0, 0])
    def test_dropout_rate_valid(self, dropout_rate):
        detector = MultilayerPerceptron(window_size=16, dropout_rate=dropout_rate)
        assert detector.dropout_rate == dropout_rate

    @pytest.mark.parametrize("dropout_rate", ["0.1", [0.2], True])
    def test_dropout_rate_invalid_type(self, dropout_rate):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize("dropout_rate", [-1.3, 1.0, 1.5])
    def test_dropout_rate_invalid_value(self, dropout_rate):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize(
        "activation_function", BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys()
    )
    def test_activation_function_valid(self, activation_function):
        detector = MultilayerPerceptron(
            window_size=16, activation_function=activation_function
        )
        assert detector.activation_function == activation_function

    @pytest.mark.parametrize("activation_function", [1, ["tanh", 1, "sigmoid"]])
    def test_activation_functions_invalid_type(self, activation_function):
        with pytest.raises(TypeError):
            MultilayerPerceptron(
                window_size=16, activation_function=activation_function
            )

    @pytest.mark.parametrize("activation_function", ["invalid-value"])
    def test_activation_functions_invalid_value(self, activation_function):
        with pytest.raises(ValueError):
            MultilayerPerceptron(
                window_size=16, activation_function=activation_function
            )

    @pytest.mark.parametrize("normalization", [True, False, [True, False, False]])
    def test_batch_normalization_valid(self, normalization):
        detector = MultilayerPerceptron(
            window_size=16, encoder_batch_normalization=normalization
        )
        assert detector.encoder_batch_normalization == normalization

    @pytest.mark.parametrize("batch_normalization", [True, False])
    def test_batch_normalization_valid(self, batch_normalization):
        detector = MultilayerPerceptron(
            window_size="fft", batch_normalization=batch_normalization
        )
        assert detector.batch_normalization == batch_normalization

    @pytest.mark.parametrize("batch_normalization", [5, 1.0, "invalid"])
    def test_batch_normalization_valid(self, batch_normalization):
        with pytest.raises(TypeError):
            MultilayerPerceptron(
                window_size="fft", batch_normalization=batch_normalization
            )


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
            dropout_rate=0,  # For simpler testing
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

    @pytest.mark.parametrize(
        "activation_function", BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys()
    )
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
        detector = MultilayerPerceptron(window_size=16, dropout_rate=0)
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
