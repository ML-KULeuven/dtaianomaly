
import textwrap
import pytest
from dtaianomaly.anomaly_detection import AutoEncoder, Supervision


class TestAutoEncoder:

    def test_supervision(self):
        assert AutoEncoder(window_size=100).supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(AutoEncoder(window_size=100)) == "AutoEncoder(window_size=100)"
        assert str(AutoEncoder(window_size=100, encoder_hidden_layer_dimension=[64])) == "AutoEncoder(window_size=100,encoder_hidden_layer_dimension=[64])"
        assert str(AutoEncoder(window_size=100, encoder_dropout_rate=0.5)) == "AutoEncoder(window_size=100,encoder_dropout_rate=0.5)"


class TestInitialization:

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


class TestArchitecture:

    def test_default_architecture(self):
        architecture = AutoEncoder(window_size=16)._initialize_architecture(64)
        expected = """
            _AutoEncoderArchitecture(
              (encoder): Sequential(
                (linear-0 (64->64)): Linear(in_features=64, out_features=64, bias=True)
                (activation-0-relu): ReLU()
                (dropout-0): Dropout(p=0.2, inplace=False)
                (linear-1 (64->32)): Linear(in_features=64, out_features=32, bias=True)
                (batch-norm-1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-1-relu): ReLU()
                (dropout-1): Dropout(p=0.2, inplace=False)
                (linear-2 (32->8)): Linear(in_features=32, out_features=8, bias=True)
                (batch-norm-2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-2-relu): ReLU()
                (dropout-2): Dropout(p=0.2, inplace=False)
              )
              (decoder): Sequential(
                (linear-0 (8->32)): Linear(in_features=8, out_features=32, bias=True)
                (batch-norm-0): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-0-relu): ReLU()
                (dropout-0): Dropout(p=0.2, inplace=False)
                (linear-1 (32->64)): Linear(in_features=32, out_features=64, bias=True)
                (batch-norm-1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-1-relu): ReLU()
                (dropout-1): Dropout(p=0.2, inplace=False)
                (linear-2 (64->64)): Linear(in_features=64, out_features=64, bias=True)
                (activation-2-relu): ReLU()
              )
            )
        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_non_symmetric_architecture(self):  # TODO rerun when the other tests are done
        architecture = AutoEncoder(window_size=16, encoder_hidden_layer_dimension=[32, 16, 8], decoder_hidden_layer_dimension=[16])._initialize_architecture(64)
        expected = """
            _AutoEncoderArchitecture(
              (encoder): Sequential(
                (linear-0 (64->32)): Linear(in_features=64, out_features=64, bias=True)
                (activation-0-relu): ReLU()
                (dropout-0): Dropout(p=0.2, inplace=False)
                (linear-1 (32->16)): Linear(in_features=64, out_features=32, bias=True)
                (batch-norm-1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-1-relu): ReLU()
                (dropout-1): Dropout(p=0.2, inplace=False)
                (linear-2 (16->8)): Linear(in_features=32, out_features=8, bias=True)
                (batch-norm-2): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-2-relu): ReLU()
                (dropout-2): Dropout(p=0.2, inplace=False)
                (linear-3 (8->8)): Linear(in_features=32, out_features=8, bias=True)
                (batch-norm-3): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-3-relu): ReLU()
                (dropout-3): Dropout(p=0.2, inplace=False)
              )
              (decoder): Sequential(
                (linear-0 (8->16)): Linear(in_features=8, out_features=32, bias=True)
                (batch-norm-0): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (activation-0-relu): ReLU()
                (dropout-0): Dropout(p=0.2, inplace=False)
                (linear-1 (16->64)): Linear(in_features=32, out_features=64, bias=True)
                (activation-1-relu): ReLU()
              )
            )
        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_single_activation_function(self):
        architecture = AutoEncoder(window_size=16, encoder_activation_functions='relu')._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_multiple_activation_functions(self):
        architecture = AutoEncoder(window_size=16, encoder_activation_functions=['relu', 'tanh', 'sigmoid'])._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_custom_decoder_single_activation_functions(self):
        architecture = AutoEncoder(window_size=16, decoder_activation_functions='relu')._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_custom_decoder_multiple_activation_functions(self):
        architecture = AutoEncoder(window_size=16, decoder_activation_functions=['relu', 'tanh', 'sigmoid'])._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_single_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, encoder_batch_normalization=False)._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_multiple_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, encoder_batch_normalization=[False, False, True])._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_custom_decoder_single_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, decoder_batch_normalization=False)._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_custom_decoder_multiple_batch_normalization(self):
        architecture = AutoEncoder(window_size=16, decoder_batch_normalization=[False, False, True])._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_single_dropout(self):
        architecture = AutoEncoder(window_size=16, encoder_dropout_rate=0.5)._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_multiple_dropout(self):
        architecture = AutoEncoder(window_size=16, encoder_dropout_rate=[0.2, 0.3, 0.5])._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_custom_decoder_single_dropout(self):
        architecture = AutoEncoder(window_size=16, decoder_dropout_rate=0.5)._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()

    def test_custom_decoder_multiple_dropout(self):
        architecture = AutoEncoder(window_size=16, decoder_dropout_rate=[0.2, 0.3, 0.5])._initialize_architecture(64)
        expected = """

        """
        assert str(list(architecture.modules())[0]) == textwrap.dedent(expected).strip()
