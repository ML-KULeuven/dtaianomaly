import pytest
from conftest import is_flatten, is_linear, is_lstm

from dtaianomaly.anomaly_detection import LongShortTermMemoryNetwork
from dtaianomaly.anomaly_detection._LongShortTermMemoryNetwork import _LSTM


class TestInitialize:

    def test_hidden_units_valid(self):
        detector = LongShortTermMemoryNetwork(window_size=16, hidden_units=4)
        assert detector.hidden_units == 4

    def test_hidden_units_invalid_type(self):
        with pytest.raises(TypeError):
            LongShortTermMemoryNetwork(window_size=16, hidden_units=32.0)

    def test_hidden_units_invalid_value(self):
        with pytest.raises(ValueError):
            LongShortTermMemoryNetwork(window_size=16, hidden_units=0)

    def test_num_lstm_layers_valid(self):
        detector = LongShortTermMemoryNetwork(window_size=16, num_lstm_layers=4)
        assert detector.num_lstm_layers == 4

    def test_num_lstm_layers_invalid_type(self):
        with pytest.raises(TypeError):
            LongShortTermMemoryNetwork(window_size=16, num_lstm_layers=32.0)

    def test_num_lstm_layers_invalid_value(self):
        with pytest.raises(ValueError):
            LongShortTermMemoryNetwork(window_size=16, num_lstm_layers=0)


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

    @pytest.mark.parametrize(
        "forecast_length,hidden_units,num_lstm_layers,bias,dropout_rate",
        [
            (1, 1, 1, False, 0.0),
            (4, 8, 1, True, 0.0),
            (8, 4, 4, False, 0.2),
            (1, 8, 8, True, 0.2),
        ],
    )
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
