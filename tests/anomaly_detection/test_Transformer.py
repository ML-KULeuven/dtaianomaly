
import pytest
import torch.nn

from dtaianomaly.anomaly_detection import Transformer, BaseNeuralDetector, Supervision
from dtaianomaly.anomaly_detection.Transformer import _adjust_nhead
from conftest import (
    is_sequential,
    is_dropout,
    is_linear,
    is_activation,
    is_flatten,
    is_transformer_encoder,
    is_transformer_encoder_layer,
    is_multihead_attention,
    is_non_dynamically_quantizable_linear,
    is_normalization
)


class TestTransformer:

    def test_supervision(self):
        assert Transformer(window_size=100).supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(Transformer(window_size=100)) == "Transformer(window_size=100)"
        assert str(Transformer(window_size=100, num_heads=17)) == "Transformer(window_size=100,num_heads=17)"
        assert str(Transformer(window_size=100, dimension_feedforward=50)) == "Transformer(window_size=100,dimension_feedforward=50)"


class TestInitialize:

    @pytest.mark.parametrize('num_heads', [4, 8, 16, 32])
    def test_num_heads_valid(self, num_heads):
        detector = Transformer(window_size=16, num_heads=num_heads)
        assert detector.num_heads == num_heads

    @pytest.mark.parametrize('num_heads', [[32], '32', 32.0, True])
    def test_num_heads_invalid_type(self, num_heads):
        with pytest.raises(TypeError):
            Transformer(window_size=16, num_heads=num_heads)

    @pytest.mark.parametrize('num_heads', [0, -1])
    def test_num_heads_invalid_value(self, num_heads):
        with pytest.raises(ValueError):
            Transformer(window_size=16, num_heads=num_heads)

    @pytest.mark.parametrize('num_transformer_layers', [4, 8, 16, 32])
    def test_num_transformer_layers_valid(self, num_transformer_layers):
        detector = Transformer(window_size=16, num_transformer_layers=num_transformer_layers)
        assert detector.num_transformer_layers == num_transformer_layers

    @pytest.mark.parametrize('num_transformer_layers', [[32], '32', 32.0, True])
    def test_num_transformer_layers_invalid_type(self, num_transformer_layers):
        with pytest.raises(TypeError):
            Transformer(window_size=16, num_transformer_layers=num_transformer_layers)

    @pytest.mark.parametrize('num_transformer_layers', [0, -1])
    def test_num_transformer_layers_invalid_value(self, num_transformer_layers):
        with pytest.raises(ValueError):
            Transformer(window_size=16, num_transformer_layers=num_transformer_layers)

    @pytest.mark.parametrize('dimension_feedforward', [4, 8, 16, 32])
    def test_dimension_feedforward_valid(self, dimension_feedforward):
        detector = Transformer(window_size=16, dimension_feedforward=dimension_feedforward)
        assert detector.dimension_feedforward == dimension_feedforward

    @pytest.mark.parametrize('dimension_feedforward', [[32], '32', 32.0, True])
    def test_dimension_feedforward_invalid_type(self, dimension_feedforward):
        with pytest.raises(TypeError):
            Transformer(window_size=16, dimension_feedforward=dimension_feedforward)

    @pytest.mark.parametrize('dimension_feedforward', [0, -1])
    def test_dimension_feedforward_invalid_value(self, dimension_feedforward):
        with pytest.raises(ValueError):
            Transformer(window_size=16, dimension_feedforward=dimension_feedforward)

    @pytest.mark.parametrize('bias', [True, False])
    def test_bias_valid(self, bias):
        detector = Transformer(window_size=16, bias=bias)
        assert detector.bias == bias

    @pytest.mark.parametrize('bias', ['True', 1, 1.0])
    def test_bias_invalid_type(self, bias):
        with pytest.raises(TypeError):
            Transformer(window_size=16, bias=bias)

    @pytest.mark.parametrize('dropout_rate', [0.1, 0.5, 0.0, 0])
    def test_dropout_rate_valid(self, dropout_rate):
        detector = Transformer(window_size=16, dropout_rate=dropout_rate)
        assert detector.dropout_rate == dropout_rate

    @pytest.mark.parametrize('dropout_rate', ['0.1', [0.2], True])
    def test_dropout_rate_invalid_type(self, dropout_rate):
        with pytest.raises(TypeError):
            Transformer(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize('dropout_rate', [-1.3, 1.0, 1.5])
    def test_dropout_rate_invalid_value(self, dropout_rate):
        with pytest.raises(ValueError):
            Transformer(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize('activation_function', BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys())
    def test_activation_function_valid(self, activation_function):
        detector = Transformer(window_size=16, activation_function=activation_function)
        assert detector.activation_function == activation_function

    @pytest.mark.parametrize('activation_function', [1, ['tanh', 1, 'sigmoid']])
    def test_activation_functions_invalid_type(self, activation_function):
        with pytest.raises(TypeError):
            Transformer(window_size=16, activation_function=activation_function)

    @pytest.mark.parametrize('activation_function', ['invalid-value'])
    def test_activation_functions_invalid_value(self, activation_function):
        with pytest.raises(ValueError):
            Transformer(window_size=16, activation_function=activation_function)


class TestBuildArchitecture:

    def test_default(self):
        detector = Transformer(window_size=16)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_transformer_encoder(next(modules), 1, True)
        assert isinstance(next(modules), torch.nn.ModuleList)

        assert is_transformer_encoder_layer(next(modules))
        assert is_multihead_attention(next(modules), 8)
        assert is_non_dynamically_quantizable_linear(next(modules), 16*8, 16*8)
        assert is_linear(next(modules), 16*8, 32)
        assert is_dropout(next(modules), 0)
        assert is_linear(next(modules), 32, 16*8)
        assert is_normalization(next(modules), (16*8,))
        assert is_normalization(next(modules), (16*8,))
        assert is_dropout(next(modules), 0)
        assert is_dropout(next(modules), 0)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 16*8, 8)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('n_attributes,window_size,num_heads,actual_num_heads,enable_tensor', [
        (4, 16, 8, 8, True),
        (6, 10, 7, 6, True),
        (7, 9, 8, 7, False),
    ])
    def test_custom_nhead(self, n_attributes, window_size, num_heads, actual_num_heads, enable_tensor):
        detector = Transformer(window_size=window_size, num_heads=num_heads)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(n_attributes).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_transformer_encoder(next(modules), 1, enable_tensor)
        assert isinstance(next(modules), torch.nn.ModuleList)

        assert is_transformer_encoder_layer(next(modules))
        assert is_multihead_attention(next(modules), actual_num_heads)
        assert is_non_dynamically_quantizable_linear(next(modules), window_size * n_attributes, window_size * n_attributes)
        assert is_linear(next(modules), window_size * n_attributes, 32)
        assert is_dropout(next(modules), 0)
        assert is_linear(next(modules), 32, window_size * n_attributes)
        assert is_normalization(next(modules), (window_size * n_attributes,))
        assert is_normalization(next(modules), (window_size * n_attributes,))
        assert is_dropout(next(modules), 0)
        assert is_dropout(next(modules), 0)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), window_size * n_attributes, n_attributes)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize('forecast_length', [4, 8])
    @pytest.mark.parametrize('num_transformer_layers', [4, 8])
    @pytest.mark.parametrize('dimension_feedforward', [64, 128])
    @pytest.mark.parametrize('bias', [True, False])
    @pytest.mark.parametrize('dropout_rate', [0.2])
    @pytest.mark.parametrize('activation_function', BaseNeuralDetector._ACTIVATION_FUNCTIONS.keys())
    def test_custom(self, forecast_length, num_transformer_layers, dimension_feedforward, bias, dropout_rate, activation_function):
        detector = Transformer(
            window_size=16,
            forecast_length=forecast_length,
            num_transformer_layers=num_transformer_layers,
            dimension_feedforward=dimension_feedforward,
            bias=bias,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_transformer_encoder(next(modules), num_transformer_layers, True)
        assert isinstance(next(modules), torch.nn.ModuleList)

        for _ in range(num_transformer_layers):
            assert is_transformer_encoder_layer(next(modules))
            assert is_multihead_attention(next(modules), 8)
            assert is_non_dynamically_quantizable_linear(next(modules), 16*8, 16*8)
            assert is_linear(next(modules), 16*8, dimension_feedforward)
            assert is_dropout(next(modules), dropout_rate)
            assert is_linear(next(modules), dimension_feedforward, 16*8)
            assert is_normalization(next(modules), (16*8,))
            assert is_normalization(next(modules), (16*8,))
            assert is_dropout(next(modules), dropout_rate)
            assert is_dropout(next(modules), dropout_rate)
            assert is_activation(next(modules), activation_function)

        assert is_linear(next(modules), 16*8, 8*forecast_length)

        with pytest.raises(StopIteration):
            next(modules)


class TestAdjustNhead:

    @pytest.mark.parametrize('d_model,nhead,expected', [
        (64, 8, 8),  # already valid
        (60, 7, 6),  # 7 is invalid, 6 is closest valid below
        (60, 8, 6),  # 8 is invalid, 10 is closest valid above
        (63, 8, 7),  # 8 is invalid, 7 is closest valid below
        (100, 3, 2),  # both 2 and 4 are valid, but prefer smaller value
        (100, 6, 5),  # closest valid is 5
        (17, 8, 1),  # prime number d_model
        (120, 15, 15),  # already valid
        (128, 12, 8),
        (5, 3, 1),
    ])
    def test(self, d_model, nhead, expected):
        assert _adjust_nhead(d_model, nhead) == expected
