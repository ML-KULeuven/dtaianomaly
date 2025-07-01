
import pytest

from dtaianomaly.anomaly_detection import AutoEncoder, BaseNeuralDetector, Supervision
from dtaianomaly.anomaly_detection.AutoEncoder import _AutoEncoderArchitecture
from conftest import (
    is_sequential,
    is_activation,
    is_batch_normalization,
    is_dropout,
    is_linear,
    is_flatten,
    is_un_flatten
)


class TestAutoEncoder:

    def test_supervision(self):
        assert AutoEncoder(window_size=100).supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(AutoEncoder(window_size=100)) == "AutoEncoder(window_size=100)"
        assert str(AutoEncoder(window_size=100, encoder_dimensions=[64, 32])) == "AutoEncoder(window_size=100,encoder_dimensions=[64, 32])"
        assert str(AutoEncoder(window_size=100, dropout_rate=0.5)) == "AutoEncoder(window_size=100,dropout_rate=0.5)"


class TestInitialization:

    @pytest.mark.parametrize('dimensions', [[32], [32, 16], [32, 16, 8], [8, 16, 32]])
    def test_encoder_dimensions_valid(self, dimensions):
        detector = AutoEncoder(window_size=16, encoder_dimensions=dimensions)
        assert detector.encoder_dimensions == dimensions

    @pytest.mark.parametrize('dimensions', [32, ['32', 16, 8]])
    def test_encoder_dimensions_invalid_type(self, dimensions):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, encoder_dimensions=dimensions)

    @pytest.mark.parametrize('dimensions', [[32, -16, 8]])
    def test_encoder_dimensions_invalid_value(self, dimensions):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_dimensions=dimensions)

    @pytest.mark.parametrize('dimension', [32, 16, 8])
    def test_latent_space_valid(self, dimension):
        detector = AutoEncoder(window_size=16, latent_space_dimension=dimension)
        assert detector.latent_space_dimension == dimension

    @pytest.mark.parametrize('dimension', ['32', 16.0, True])
    def test_latent_space_invalid_type(self, dimension):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, latent_space_dimension=dimension)

    @pytest.mark.parametrize('dimension', [0, -1, -16])
    def test_latent_space_invalid_value(self, dimension):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, latent_space_dimension=dimension)

    @pytest.mark.parametrize('dimensions', [[32], [32, 16], [32, 16, 8], [8, 16, 32]])
    def test_decoder_dimensions_valid(self, dimensions):
        detector = AutoEncoder(window_size=16, decoder_dimensions=dimensions)
        assert detector.decoder_dimensions == dimensions

    @pytest.mark.parametrize('dimensions', [32, ['32', 16, 8]])
    def test_decoder_dimensions_invalid_type(self, dimensions):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, decoder_dimensions=dimensions)

    @pytest.mark.parametrize('dimensions', [[32, -16, 8]])
    def test_decoder_dimensions_invalid_value(self, dimensions):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_dimensions=dimensions)

    @pytest.mark.parametrize('dropout_rate', [0.1, 0.5, 0.0, 0])
    def test_dropout_rate_valid(self, dropout_rate):
        detector = AutoEncoder(window_size=16, dropout_rate=dropout_rate)
        assert detector.dropout_rate == dropout_rate

    @pytest.mark.parametrize('dropout_rate', ['0.1', [0.2], True])
    def test_dropout_rate_invalid_type(self, dropout_rate):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize('dropout_rate', [-1.3, 1.0, 1.5])
    def test_dropout_rate_invalid_value(self, dropout_rate):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize('activation_function', BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys())
    def test_activation_function_valid(self, activation_function):
        detector = AutoEncoder(window_size=16, activation_function=activation_function)
        assert detector.activation_function == activation_function

    @pytest.mark.parametrize('activation_function', [1, ['tanh', 1, 'sigmoid']])
    def test_activation_functions_invalid_type(self, activation_function):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, activation_function=activation_function)

    @pytest.mark.parametrize('activation_function', ['invalid-value'])
    def test_activation_functions_invalid_value(self, activation_function):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, activation_function=activation_function)

    @pytest.mark.parametrize('normalization', [True, False, [True, False, False]])
    def test_batch_normalization_valid(self, normalization):
        detector = AutoEncoder(window_size=16, encoder_batch_normalization=normalization)
        assert detector.encoder_batch_normalization == normalization

    @pytest.mark.parametrize('batch_normalization', [True, False])
    def test_batch_normalization_valid(self, batch_normalization):
        detector = AutoEncoder(window_size='fft', batch_normalization=batch_normalization)
        assert detector.batch_normalization == batch_normalization

    @pytest.mark.parametrize('batch_normalization', [5, 1.0, 'invalid'])
    def test_batch_normalization_valid(self, batch_normalization):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', batch_normalization=batch_normalization)


class TestBuildArchitecture:

    def test_default(self):
        detector = AutoEncoder(window_size=16)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('encoder_dimensions', [
        [],
        [64],
        [64, 32],
        [64, 32, 16]
    ])
    def test_custom_encoder_layers(self, encoder_dimensions):
        detector = AutoEncoder(
            window_size=16,
            encoder_dimensions=encoder_dimensions,
            latent_space_dimension=8,
            batch_normalization=False,  # For simpler testing
            dropout_rate=0,  # For simpler testing
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        dimensions = [128, *encoder_dimensions, 8]
        for d, d_ in zip(dimensions[:-1], dimensions[1:]):
            assert is_linear(next(modules), d, d_)
            assert is_activation(next(modules), 'relu')

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 64)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('latent_space_dimension', [2, 4, 8, 16])
    def test_latent_space_dimension(self, latent_space_dimension):
        detector = AutoEncoder(
            window_size=16,
            latent_space_dimension=latent_space_dimension
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, latent_space_dimension)
        assert is_batch_normalization(next(modules), latent_space_dimension)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), latent_space_dimension, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('decoder_dimensions', [
        [],
        [64],
        [32, 64],
        [16, 32, 64]
    ])
    def test_custom_decoder_layers(self, decoder_dimensions):
        detector = AutoEncoder(
            window_size=16,
            decoder_dimensions=decoder_dimensions,
            latent_space_dimension=8,
            batch_normalization=False,  # For simpler testing
            dropout_rate=0,  # For simpler testing
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 64, 8)
        assert is_activation(next(modules), 'relu')

        assert is_sequential(next(modules))  # Decoder

        dimensions = [8, *decoder_dimensions, 128]
        for d, d_ in zip(dimensions[:-1], dimensions[1:]):
            assert is_linear(next(modules), d, d_)
            assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_batch_normalization(self):
        detector = AutoEncoder(
            window_size=16,
            batch_normalization=False,
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()
        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 32, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('activation_function', BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys())
    def test_custom_activation_function(self, activation_function):
        detector = AutoEncoder(
            window_size=16,
            activation_function=activation_function
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), activation_function)
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), activation_function)
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), activation_function)
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), activation_function)

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('dropout_rate', [0.1, 0.3])
    def test_custom_dropout_rate(self, dropout_rate):
        detector = AutoEncoder(
            window_size=16,
            dropout_rate=dropout_rate
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), dropout_rate)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), dropout_rate)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), dropout_rate)

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)

    def test_zero_dropout_rate(self):
        detector = AutoEncoder(
            window_size=16,
            dropout_rate=0
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)

        assert is_sequential(next(modules))  # Encoder
        assert is_flatten(next(modules))

        assert is_linear(next(modules), 128, 64)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 64, 128)
        assert is_activation(next(modules), 'relu')

        assert is_un_flatten(next(modules))

        with pytest.raises(StopIteration):
            next(modules)
