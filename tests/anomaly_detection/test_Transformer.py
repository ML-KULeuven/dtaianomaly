import pytest
import torch.nn
from conftest import (
    is_activation,
    is_dropout,
    is_flatten,
    is_linear,
    is_multihead_attention,
    is_non_dynamically_quantizable_linear,
    is_normalization,
    is_sequential,
    is_transformer_encoder,
    is_transformer_encoder_layer,
)

from dtaianomaly.anomaly_detection import Supervision, Transformer
from dtaianomaly.anomaly_detection._Transformer import _adjust_nhead


class TestTransformer:

    def test_supervision(self):
        assert Transformer(window_size=100).supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(Transformer(window_size=100)) == "Transformer(window_size=100)"
        assert (
            str(Transformer(window_size=100, num_heads=17))
            == "Transformer(window_size=100,num_heads=17)"
        )
        assert (
            str(Transformer(window_size=100, dimension_feedforward=50))
            == "Transformer(window_size=100,dimension_feedforward=50)"
        )


class TestInitialize:

    def test_num_heads_valid(self):
        detector = Transformer(window_size=16, num_heads=4)
        assert detector.num_heads == 4

    def test_num_heads_invalid_type(self):
        with pytest.raises(TypeError):
            Transformer(window_size=16, num_heads=32.0)

    def test_num_heads_invalid_value(self):
        with pytest.raises(ValueError):
            Transformer(window_size=16, num_heads=0)

    def test_num_transformer_layers_valid(self):
        detector = Transformer(window_size=16, num_transformer_layers=4)
        assert detector.num_transformer_layers == 4

    def test_num_transformer_layers_invalid_type(self):
        with pytest.raises(TypeError):
            Transformer(window_size=16, num_transformer_layers=32.0)

    def test_num_transformer_layers_invalid_value(self):
        with pytest.raises(ValueError):
            Transformer(window_size=16, num_transformer_layers=0)

    def test_dimension_feedforward_valid(self):
        detector = Transformer(window_size=16, dimension_feedforward=4)
        assert detector.dimension_feedforward == 4

    def test_dimension_feedforward_invalid_type(self):
        with pytest.raises(TypeError):
            Transformer(window_size=16, dimension_feedforward=32.0)

    def test_dimension_feedforward_invalid_value(self):
        with pytest.raises(ValueError):
            Transformer(window_size=16, dimension_feedforward=0)


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
        assert is_non_dynamically_quantizable_linear(next(modules), 16 * 8, 16 * 8)
        assert is_linear(next(modules), 16 * 8, 32)
        assert is_dropout(next(modules), 0)
        assert is_linear(next(modules), 32, 16 * 8)
        assert is_normalization(next(modules), (16 * 8,))
        assert is_normalization(next(modules), (16 * 8,))
        assert is_dropout(next(modules), 0)
        assert is_dropout(next(modules), 0)
        assert is_activation(next(modules), "relu")

        assert is_linear(next(modules), 16 * 8, 8)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize(
        "n_attributes,window_size,num_heads,actual_num_heads,enable_tensor",
        [
            (4, 16, 8, 8, True),
            (6, 10, 7, 6, True),
            (7, 9, 8, 7, False),
        ],
    )
    def test_custom_nhead(
        self, n_attributes, window_size, num_heads, actual_num_heads, enable_tensor
    ):
        detector = Transformer(window_size=window_size, num_heads=num_heads)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(n_attributes).modules()

        assert is_sequential(next(modules))
        assert is_flatten(next(modules))

        assert is_transformer_encoder(next(modules), 1, enable_tensor)
        assert isinstance(next(modules), torch.nn.ModuleList)

        assert is_transformer_encoder_layer(next(modules))
        assert is_multihead_attention(next(modules), actual_num_heads)
        assert is_non_dynamically_quantizable_linear(
            next(modules), window_size * n_attributes, window_size * n_attributes
        )
        assert is_linear(next(modules), window_size * n_attributes, 32)
        assert is_dropout(next(modules), 0)
        assert is_linear(next(modules), 32, window_size * n_attributes)
        assert is_normalization(next(modules), (window_size * n_attributes,))
        assert is_normalization(next(modules), (window_size * n_attributes,))
        assert is_dropout(next(modules), 0)
        assert is_dropout(next(modules), 0)
        assert is_activation(next(modules), "relu")

        assert is_linear(next(modules), window_size * n_attributes, n_attributes)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize(
        "forecast_length,num_transformer_layers,dimension_feedforward,bias,dropout_rate,activation_function",
        [
            (4, 1, 64, True, 0.2, "linear"),
            (8, 4, 128, False, 0.2, "relu"),
            (1, 2, 128, True, 0.0, "sigmoid"),
            (4, 8, 64, False, 0.2, "tanh"),
        ],
    )
    def test_custom(
        self,
        forecast_length,
        num_transformer_layers,
        dimension_feedforward,
        bias,
        dropout_rate,
        activation_function,
    ):
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
            assert is_non_dynamically_quantizable_linear(next(modules), 16 * 8, 16 * 8)
            assert is_linear(next(modules), 16 * 8, dimension_feedforward)
            assert is_dropout(next(modules), dropout_rate)
            assert is_linear(next(modules), dimension_feedforward, 16 * 8)
            assert is_normalization(next(modules), (16 * 8,))
            assert is_normalization(next(modules), (16 * 8,))
            assert is_dropout(next(modules), dropout_rate)
            assert is_dropout(next(modules), dropout_rate)
            assert is_activation(next(modules), activation_function)

        assert is_linear(next(modules), 16 * 8, 8 * forecast_length)

        with pytest.raises(StopIteration):
            next(modules)


class TestAdjustNhead:

    @pytest.mark.parametrize(
        "d_model,nhead,expected",
        [
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
        ],
    )
    def test(self, d_model, nhead, expected):
        assert _adjust_nhead(d_model, nhead) == expected
