import pytest
from conftest import is_flatten, is_linear, is_lstm

from dtaianomaly.anomaly_detection import LongShortTermMemoryNetwork, Supervision
from dtaianomaly.anomaly_detection._LongShortTermMemoryNetwork import _LSTM


class TestInitialize:

    @pytest.mark.parametrize("hidden_units", [4, 8, 16, 32])
    def test_hidden_units_valid(self, hidden_units):
        detector = LongShortTermMemoryNetwork(window_size=16, hidden_units=hidden_units)
        assert detector.hidden_units == hidden_units

    @pytest.mark.parametrize("hidden_units", [[32], "32", 32.0, True])
    def test_hidden_units_invalid_type(self, hidden_units):
        with pytest.raises(TypeError):
            LongShortTermMemoryNetwork(window_size=16, hidden_units=hidden_units)

    @pytest.mark.parametrize("hidden_units", [0, -1])
    def test_hidden_units_invalid_value(self, hidden_units):
        with pytest.raises(ValueError):
            LongShortTermMemoryNetwork(window_size=16, hidden_units=hidden_units)

    @pytest.mark.parametrize("num_lstm_layers", [4, 8, 16, 32])
    def test_num_lstm_layers_valid(self, num_lstm_layers):
        detector = LongShortTermMemoryNetwork(
            window_size=16, num_lstm_layers=num_lstm_layers
        )
        assert detector.num_lstm_layers == num_lstm_layers

    @pytest.mark.parametrize("num_lstm_layers", [[32], "32", 32.0, True])
    def test_num_lstm_layers_invalid_type(self, num_lstm_layers):
        with pytest.raises(TypeError):
            LongShortTermMemoryNetwork(window_size=16, num_lstm_layers=num_lstm_layers)

    @pytest.mark.parametrize("num_lstm_layers", [0, -1])
    def test_num_lstm_layers_invalid_value(self, num_lstm_layers):
        with pytest.raises(ValueError):
            LongShortTermMemoryNetwork(window_size=16, num_lstm_layers=num_lstm_layers)

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_valid(self, bias):
        detector = LongShortTermMemoryNetwork(window_size=16, bias=bias)
        assert detector.bias == bias

    @pytest.mark.parametrize("bias", ["True", 1, 1.0])
    def test_bias_invalid_type(self, bias):
        with pytest.raises(TypeError):
            LongShortTermMemoryNetwork(window_size=16, bias=bias)

    @pytest.mark.parametrize("dropout_rate", [0.1, 0.5, 0.0])
    def test_dropout_rate_valid(self, dropout_rate):
        detector = LongShortTermMemoryNetwork(window_size=16, dropout_rate=dropout_rate)
        assert detector.dropout_rate == dropout_rate

    @pytest.mark.parametrize("dropout_rate", ["0.1", [0.2], True])
    def test_dropout_rate_invalid_type(self, dropout_rate):
        with pytest.raises(TypeError):
            LongShortTermMemoryNetwork(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize("dropout_rate", [-1.3, 1.0, 1.5])
    def test_dropout_rate_invalid_value(self, dropout_rate):
        with pytest.raises(ValueError):
            LongShortTermMemoryNetwork(window_size=16, dropout_rate=dropout_rate)


class TestBuildArchitecture:

    def test_default(self):
        detector = LongShortTermMemoryNetwork(window_size=16)
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _LSTM)

        assert is_lstm(next(modules), 8, 1 * 8, 1, True, 0.0)
        assert is_flatten(next(modules))
        assert is_linear(next(modules), 16 * 8, 8)

        with pytest.raises(StopIteration):
            next(modules)

    @pytest.mark.parametrize("forecast_length", [1, 4, 8])
    @pytest.mark.parametrize("hidden_units", [1, 4, 8])
    @pytest.mark.parametrize("num_lstm_layers", [1, 4, 8])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
    def test_custom(
        self, forecast_length, hidden_units, num_lstm_layers, bias, dropout_rate
    ):
        detector = LongShortTermMemoryNetwork(
            window_size=16,
            forecast_length=forecast_length,
            hidden_units=hidden_units,
            num_lstm_layers=num_lstm_layers,
            bias=bias,
            dropout_rate=dropout_rate,
        )
        detector.window_size_ = detector.window_size
        modules = detector._build_architecture(8).modules()

        assert isinstance(next(modules), _LSTM)

        assert is_lstm(
            next(modules),
            8,
            forecast_length * hidden_units,
            num_lstm_layers,
            bias,
            dropout_rate,
        )
        assert is_flatten(next(modules))
        assert is_linear(
            next(modules), 16 * forecast_length * hidden_units, 8 * forecast_length
        )

        with pytest.raises(StopIteration):
            next(modules)
