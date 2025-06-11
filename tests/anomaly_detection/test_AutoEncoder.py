
import pytest
import numpy as np
import torch

from dtaianomaly.anomaly_detection import AutoEncoder, Supervision
from dtaianomaly.anomaly_detection.AutoEncoder import _AutoEncoderArchitecture
from conftest import (
    is_sequential,
    is_activation,
    is_batch_normalization,
    is_dropout,
    is_linear
)

_VALID_ERROR_METRICS = ['mae', 'mse', 'l1', 'l2']


class TestAutoEncoder:

    def test_supervision(self):
        assert AutoEncoder(window_size=100).supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(AutoEncoder(window_size=100)) == "AutoEncoder(window_size=100)"
        assert str(AutoEncoder(window_size=100, encoder_hidden_layer_dimension=[64])) == "AutoEncoder(window_size=100,encoder_hidden_layer_dimension=[64])"
        assert str(AutoEncoder(window_size=100, encoder_dropout_rate=0.5)) == "AutoEncoder(window_size=100,encoder_dropout_rate=0.5)"


class TestInitialization:

    @pytest.mark.parametrize('error_metric', _VALID_ERROR_METRICS)
    def test_error_metric_valid(self, error_metric):
        detector = AutoEncoder(window_size=16, error_metric=error_metric)
        assert detector.error_metric == error_metric

    @pytest.mark.parametrize('error_metric', [32, 16.0, True])
    def test_error_metric_invalid_type(self, error_metric):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, error_metric=error_metric)

    @pytest.mark.parametrize('error_metric', ['invalid'])
    def test_error_metric_invalid_value(self, error_metric):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, error_metric=error_metric)

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
    def test_hidden_layer_dimension_valid(self, dimensions):
        detector = AutoEncoder(window_size=16, encoder_hidden_layer_dimension=dimensions)
        assert detector.encoder_hidden_layer_dimension == dimensions
        detector = AutoEncoder(window_size=16, decoder_hidden_layer_dimension=dimensions)
        assert detector.decoder_hidden_layer_dimension == dimensions

    @pytest.mark.parametrize('dimensions', [32, ['32', 16, 8], [32, 16, True]])
    def test_hidden_layer_dimension_invalid_type(self, dimensions):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, encoder_hidden_layer_dimension=dimensions)
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, decoder_hidden_layer_dimension=dimensions)

    @pytest.mark.parametrize('dimensions', [[32, -16, 8]])
    def test_hidden_layer_dimension_invalid_value(self, dimensions):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_hidden_layer_dimension=dimensions)
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_hidden_layer_dimension=dimensions)

    @pytest.mark.parametrize('activations', ['relu', 'tanh', ['tanh', 'relu', 'sigmoid']])
    def test_activation_functions_valid(self, activations):
        detector = AutoEncoder(window_size=16, encoder_activation_functions=activations)
        assert detector.encoder_activation_functions == activations
        detector = AutoEncoder(window_size=16, decoder_activation_functions=activations)
        assert detector.decoder_activation_functions == activations

    @pytest.mark.parametrize('activations', [1, ['tanh', 1, 'sigmoid']])
    def test_activation_functions_invalid_type(self, activations):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, encoder_activation_functions=activations)
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, decoder_activation_functions=activations)

    @pytest.mark.parametrize('activations', ['invalid-value', ['tanh', 'invalid-value', 'sigmoid']])
    def test_activation_functions_invalid_value(self, activations):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_activation_functions=activations)
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_activation_functions=activations)

    def test_activation_functions_invalid_length(self):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_hidden_layer_dimension=[32], encoder_activation_functions=['relu', 'tanh', 'linear'])
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_hidden_layer_dimension=[32], decoder_activation_functions=['relu', 'tanh', 'linear'])

    @pytest.mark.parametrize('normalization', [True, False, [True, False, False]])
    def test_batch_normalization_valid(self, normalization):
        detector = AutoEncoder(window_size=16, encoder_batch_normalization=normalization)
        assert detector.encoder_batch_normalization == normalization
        detector = AutoEncoder(window_size=16, decoder_batch_normalization=normalization)
        assert detector.decoder_batch_normalization == normalization

    @pytest.mark.parametrize('normalization', ['True', 1, [True, False, 'False']])
    def test_batch_normalization_invalid_type(self, normalization):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, encoder_batch_normalization=normalization)
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, decoder_batch_normalization=normalization)

    def test_batch_normalization_invalid_length(self):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_hidden_layer_dimension=[32], encoder_batch_normalization=[True, True, True])
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_hidden_layer_dimension=[32], decoder_batch_normalization=[True, True, True])

    @pytest.mark.parametrize('dropouts', [0.1, 0.5, [0.2, 0.2, 0.1], 0.0])
    def test_dropout_rates_valid(self, dropouts):
        detector = AutoEncoder(window_size=16, encoder_dropout_rate=dropouts)
        assert detector.encoder_dropout_rate == dropouts
        detector = AutoEncoder(window_size=16, decoder_dropout_rate=dropouts)
        assert detector.decoder_dropout_rate == dropouts

    @pytest.mark.parametrize('dropouts', ['0.1', [0.2, '0.2', 0.1]])
    def test_dropout_rates_invalid_type(self, dropouts):
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, encoder_dropout_rate=dropouts)
        with pytest.raises(TypeError):
            AutoEncoder(window_size=16, decoder_dropout_rate=dropouts)

    @pytest.mark.parametrize('dropouts', [-1.3, 1.0, 1.5, [0.2, -0.2, 0.1]])
    def test_dropout_rates_invalid_value(self, dropouts):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_dropout_rate=dropouts)
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_dropout_rate=dropouts)

    def test_dropout_rates_invalid_length(self):
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, encoder_hidden_layer_dimension=[32], encoder_dropout_rate=[0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16, decoder_hidden_layer_dimension=[32], decoder_dropout_rate=[0.2, 0.3, 0.4])


class TestErrorMetric:

    @pytest.mark.parametrize('error_metric', _VALID_ERROR_METRICS)
    def test(self, error_metric, univariate_time_series):
        auto_encoder = AutoEncoder(window_size=12, error_metric=error_metric, n_epochs=1)
        auto_encoder.fit(univariate_time_series)
        auto_encoder.decision_function(univariate_time_series)

    def test_update_after_initialization(self, univariate_time_series):
        auto_encoder = AutoEncoder(window_size=12, error_metric='mae', n_epochs=1)
        auto_encoder.fit(univariate_time_series)

        auto_encoder.error_metric = 'invalid'
        with pytest.raises(ValueError):
            auto_encoder.decision_function(univariate_time_series)


class TestInitializeDataset:

    @pytest.mark.parametrize('seed', [0])
    def test(self, seed):
        rng = np.random.default_rng(seed)
        t = rng.integers(100, 1000)
        n = rng.integers(1, 10)
        X = rng.uniform(size=(t, n))

        dataset = AutoEncoder(window_size=1)._initialize_dataset(X)
        assert len(dataset) == t
        for i in range(len(dataset)):
            (data, target) = dataset[i]
            assert torch.equal(data, target)
            assert torch.equal(data, torch.from_numpy(X[i]))


class TestArchitecture:

    def test_default_architecture(self):
        architecture = AutoEncoder(window_size=16)._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_non_symmetric_architecture(self):
        architecture = AutoEncoder(window_size=16, encoder_hidden_layer_dimension=[32, 16, 8], decoder_hidden_layer_dimension=[16])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 16)
        assert is_batch_normalization(next(modules), 16)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 16, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 16)
        assert is_batch_normalization(next(modules), 16)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 16, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_single_activation_function(self):
        architecture = AutoEncoder(window_size=16, encoder_activation_functions='tanh')._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'tanh')

        with pytest.raises(StopIteration):
            next(modules)

    def test_multiple_activation_functions(self):
        architecture = AutoEncoder(window_size=16, encoder_activation_functions=['relu', 'tanh', 'sigmoid'])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'sigmoid')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'sigmoid')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_custom_decoder_single_activation_functions(self):
        architecture = AutoEncoder(window_size=16, decoder_activation_functions='tanh')._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'tanh')

        with pytest.raises(StopIteration):
            next(modules)

    def test_custom_decoder_multiple_activation_functions(self):
        architecture = AutoEncoder(window_size=16, decoder_activation_functions=['relu', 'tanh', 'sigmoid'])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'sigmoid')

        with pytest.raises(StopIteration):
            next(modules)

    def test_single_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, encoder_batch_normalization=False)._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_multiple_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, encoder_batch_normalization=[False, False, True])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_custom_decoder_single_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, decoder_batch_normalization=False)._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_custom_decoder_multiple_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, decoder_batch_normalization=[False, True, False])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_single_dropout(self):
        architecture = AutoEncoder(window_size=16, encoder_dropout_rate=0.5)._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_multiple_dropout(self):
        architecture = AutoEncoder(window_size=16, encoder_dropout_rate=[0.2, 0.3, 0.5])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.3)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.3)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_custom_decoder_single_dropout(self):
        architecture = AutoEncoder(window_size=16, decoder_dropout_rate=0.5)._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.5)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_custom_decoder_multiple_dropout(self):
        architecture = AutoEncoder(window_size=16, decoder_dropout_rate=[0.2, 0.3, 0.5])._initialize_architecture(64)
        modules = architecture.modules()

        assert isinstance(next(modules), _AutoEncoderArchitecture)
        assert is_sequential(next(modules))  # Encoder

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 64, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_sequential(next(modules))  # Decoder

        assert is_linear(next(modules), 8, 32)
        assert is_batch_normalization(next(modules), 32)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 32, 64)
        assert is_batch_normalization(next(modules), 64)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.3)

        assert is_linear(next(modules), 64, 64)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)
