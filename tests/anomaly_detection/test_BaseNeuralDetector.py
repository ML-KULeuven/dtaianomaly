
import pytest
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dtaianomaly import utils
from dtaianomaly.anomaly_detection import AutoEncoder, NeuralBaseDetector
from dtaianomaly.anomaly_detection.NeuralBaseDetector import (
    _scale,
    _initialize_optimizer,
    _initialize_activation_function,
    _initialize_data_loader,
    _initialize_mlp
)
from conftest import (
    is_sequential,
    is_activation,
    is_batch_normalization,
    is_dropout,
    is_linear
)


class TestNeuralBaseDetector:

    def test_str(self):
        assert str(AutoEncoder(window_size=100)) == "AutoEncoder(window_size=100)"
        assert str(AutoEncoder(window_size=100, learning_rate=0.01)) == "AutoEncoder(window_size=100,learning_rate=0.01)"
        assert str(AutoEncoder(window_size=100, loss_function=torch.nn.L1Loss())) == "AutoEncoder(window_size=100,loss_function=L1Loss())"
        assert str(AutoEncoder(window_size=100, data_loader_kwargs={'shuffle': True})) == "AutoEncoder(window_size=100,data_loader_kwargs={'shuffle': True})"
        assert str(AutoEncoder(window_size=100, optimizer_kwargs={'weight_decay': 0.001})) == "AutoEncoder(window_size=100,optimizer_kwargs={'weight_decay': 0.001})"


class TestInitialize:

    def test_window_size_and_stride(self):
        # Quick and dirty tests as these things have generally been tested elsewhere
        AutoEncoder(window_size='fft')
        AutoEncoder(window_size='suss')
        AutoEncoder(window_size=16)
        with pytest.raises(ValueError):
            AutoEncoder(window_size=16.0)
        with pytest.raises(ValueError):
            AutoEncoder(window_size=-5)

        AutoEncoder(window_size='fft', stride=1)
        AutoEncoder(window_size='fft', stride=5)
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', stride=1.0)
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', stride=-1)

    @pytest.mark.parametrize('scaling_method', [None, 'min-max-scaling', 'standard-scaling'])
    def test_scaling_method_valid(self, scaling_method):
        detector = AutoEncoder(window_size='fft', scaling_method=scaling_method)
        assert detector.scaling_method == scaling_method

    @pytest.mark.parametrize('scaling_method', [5, 1.0, True])
    def test_scaling_method_invalid_type(self, scaling_method):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', scaling_method=scaling_method)

    @pytest.mark.parametrize('scaling_method', ['invalid-scaling'])
    def test_scaling_method_invalid_value(self, scaling_method):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', scaling_method=scaling_method)

    def test_scaling_kwargs_kwargs_valid(self):
        detector = AutoEncoder(window_size='fft', scaling_method='min-max-scaling', scaling_kwargs={'feature_range': (0, 5)})
        assert detector.scaling_kwargs == {'feature_range': (0, 5)}

    def test_scaling_kwargs_kwargs_invalid_type(self):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', scaling_method='min-max-scaling', scaling_kwargs={'invalid_kwarg': 1e-5})
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', scaling_method='min-max-scaling', scaling_kwargs='invalid')
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', scaling_method='standard-scaling', scaling_kwargs={'feature_range': (0, 5)})

    @pytest.mark.parametrize('batch_size', [8, 16, 32])
    def test_batch_size_valid(self, batch_size):
        detector = AutoEncoder(window_size='fft', batch_size=batch_size)
        assert detector.batch_size == batch_size

    @pytest.mark.parametrize('batch_size', ['8', 8.0])
    def test_batch_size_invalid_type(self, batch_size):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', batch_size=batch_size)

    @pytest.mark.parametrize('batch_size', [0, -8])
    def test_batch_size_invalid_value(self, batch_size):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', batch_size=batch_size)

    def test_data_loader_kwargs_valid(self):
        detector = AutoEncoder(window_size='fft', data_loader_kwargs={'shuffle': True})
        assert detector.data_loader_kwargs == {'shuffle': True}

    def test_data_loader_kwargs_invalid_type(self):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', data_loader_kwargs={'invalid-kwarg': True})
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', data_loader_kwargs='invalid-kwarg')

    @pytest.mark.parametrize('validation_size', [0, 0.2, 0.5])
    def test_validation_size_valid(self, validation_size):
        detector = AutoEncoder(window_size='fft', validation_size=validation_size)
        assert detector.validation_size == validation_size

    @pytest.mark.parametrize('validation_size', ['0', False])
    def test_validation_size_invalid_type(self, validation_size):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', validation_size=validation_size)

    @pytest.mark.parametrize('validation_size', [-0.2, 1.0])
    def test_validation_size_invalid_value(self, validation_size):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', validation_size=validation_size)

    @pytest.mark.parametrize('optimizer', ['Adam', 'SGD'])
    def test_optimizer_valid(self, optimizer):
        detector = AutoEncoder(window_size='fft', optimizer=optimizer)
        assert detector.optimizer == optimizer

    @pytest.mark.parametrize('optimizer', [10, False])
    def test_optimizer_invalid_type(self, optimizer):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', optimizer=optimizer)

    @pytest.mark.parametrize('optimizer', ['invalid-optimizer'])
    def test_optimizer_invalid_value(self, optimizer):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', optimizer=optimizer)

    @pytest.mark.parametrize('learning_rate', [0.01, 1.0, 10])
    def test_learning_rate_valid(self, learning_rate):
        detector = AutoEncoder(window_size='fft', learning_rate=learning_rate)
        assert detector.learning_rate == learning_rate

    @pytest.mark.parametrize('learning_rate', ['0.001', False])
    def test_learning_rate_invalid_type(self, learning_rate):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', learning_rate=learning_rate)

    @pytest.mark.parametrize('learning_rate', [-1, 0])
    def test_learning_rate_invalid_value(self, learning_rate):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', learning_rate=learning_rate)

    def test_optimizer_kwargs_kwargs_valid(self):
        detector = AutoEncoder(window_size='fft', optimizer_kwargs={'weight_decay': 1e-5})
        assert detector.optimizer_kwargs == {'weight_decay': 1e-5}

    def test_optimizer_kwargs_kwargs_invalid_type(self):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', optimizer_kwargs={'invalid_kwarg': 1e-5})
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', optimizer_kwargs='invalid')

    @pytest.mark.parametrize('n_epochs', [5, 100])
    def test_n_epochs_valid(self, n_epochs):
        detector = AutoEncoder(window_size='fft', n_epochs=n_epochs)
        assert detector.n_epochs == n_epochs

    @pytest.mark.parametrize('n_epochs', [True, '100'])
    def test_n_epochs_invalid_type(self, n_epochs):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', n_epochs=n_epochs)

    @pytest.mark.parametrize('n_epochs', [-1, 0])
    def test_n_epochs_invalid_value(self, n_epochs):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', n_epochs=n_epochs)

    @pytest.mark.parametrize('loss_function', [torch.nn.MSELoss(), torch.nn.L1Loss()])
    def test_loss_function_valid(self, loss_function):
        detector = AutoEncoder(window_size='fft', loss_function=loss_function)
        assert detector.loss_function == loss_function

    @pytest.mark.parametrize('loss_function', ['torch.nn.MSELoss()', 'MSELoss', 10])
    def test_loss_function_invalid_type(self, loss_function):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', loss_function=loss_function)

    def test_device_cpu(self):
        detector = AutoEncoder(window_size='fft', device='cpu')
        assert detector.device == 'cpu'

    def test_device_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        detector = AutoEncoder(window_size='fft', device='cuda')
        assert detector.device == 'cuda'

    def test_device_cuda_not_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', device='cuda')

    def test_device_cuda_invalid_index(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', device='cuda:10000000')

    @pytest.mark.parametrize('device', [True, 100])
    def test_device_invalid_type(self, device):
        with pytest.raises(TypeError):
            AutoEncoder(window_size='fft', device=device)

    @pytest.mark.parametrize('device', ['invalid-device'])
    def test_device_invalid_value(self, device):
        with pytest.raises(ValueError):
            AutoEncoder(window_size='fft', device=device)


class TestScaling:

    def test_no_scaling(self):
        np.random.seed(42)
        X = np.random.uniform(size=(100, 5))
        X_, scaler = _scale(X, None, None)

        assert np.array_equal(X, X_)
        assert scaler is None

    def test_standard_scaling(self):
        X = np.array([
            [5, 10, 15],
            [9, 15, 10]
        ])
        X_, scaler = _scale(X, 'standard-scaling', None)
        assert np.allclose(X_, np.array([[-1.22474487, 0, 1.22474487], [-0.88900089, 1.3970014, -0.50800051]]))
        assert isinstance(scaler, StandardScaler)

    def test_min_max_scaling(self):
        X = np.array([
            [5, 10, 15],
            [0, 75, 100]
        ])
        X_, scaler = _scale(X, 'min-max-scaling', None)
        assert np.array_equal(X_, np.array([[0, 0.5, 1], [0, 0.75, 1]]))
        assert isinstance(scaler, MinMaxScaler)

    def test_kwargs(self):
        X = np.array([
            [5, 10, 15],
            [0, 75, 100]
        ])
        X_, scaler = _scale(X, 'min-max-scaling', {'feature_range': (0, 5)})
        assert np.array_equal(X_, np.array([[0, 2.5, 5.0], [0, 3.75, 5]]))
        assert isinstance(scaler, MinMaxScaler)

    @pytest.mark.parametrize('scaling_method', [None, 'standard-scaling', 'min-max-scaling'])
    def test_training(self, univariate_time_series, scaling_method):
        auto_encoder = AutoEncoder(window_size=12, scaling_method=scaling_method, n_epochs=1)
        auto_encoder.fit(univariate_time_series)
        auto_encoder.decision_function(univariate_time_series)


class TestInitializeOptimizer:

    @pytest.mark.parametrize('optimizer,expected_cls', [
        ('Adam', torch.optim.Adam),
        ('SGD', torch.optim.SGD),
        ('Adagrad', torch.optim.Adagrad),
        ('RMSprop', torch.optim.RMSprop),
    ])
    def test(self, optimizer, expected_cls):
        optimizer = _initialize_optimizer([torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))], optimizer, 1e-3, None)
        assert isinstance(optimizer, expected_cls)

    def test_invalid_optimizer(self):
        with pytest.raises(ValueError):
            _initialize_optimizer([torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))], 'invalid', 1e-3, None)

    @pytest.mark.parametrize('learning_rate', [0.01, 0.00005])
    def test_learning_rate(self, learning_rate):
        optimizer = _initialize_optimizer([torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))], 'Adam', learning_rate, None)
        assert optimizer.param_groups[0]['lr'] == learning_rate

    @pytest.mark.parametrize('weight_decay', [0.01, 0.00005])
    def test_kwargs(self, weight_decay):
        optimizer = _initialize_optimizer([torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))], 'Adam', 1e-3, {'weight_decay': weight_decay})
        assert optimizer.param_groups[0]['weight_decay'] == weight_decay

    def test_learning_rate_with_kwargs(self):
        optimizer = _initialize_optimizer([torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))], 'Adam', 10, {'weight_decay': 0.01, 'lr': 0.01})
        assert optimizer.param_groups[0]['lr'] == 10

        optimizer = _initialize_optimizer([torch.nn.Parameter(torch.randn(3, 3, requires_grad=True))], 'Adam', 10, {'weight_decay': 0.01})
        assert optimizer.param_groups[0]['lr'] == 10


class TestInitializeDataLoader:

    def test(self):
        dataset = torch.utils.data.TensorDataset(torch.empty((10, 3)), torch.empty((10,)))
        data_loader = _initialize_data_loader(
            dataset=dataset,
            batch_size=32,
            data_loader_kwargs=None
        )
        assert data_loader.dataset == dataset
        assert data_loader.batch_size == 32

    @pytest.mark.parametrize('drop_last', [True, False])
    def test_kwargs(self, drop_last):
        data_loader = _initialize_data_loader(
            dataset=torch.utils.data.TensorDataset(torch.empty((10, 3)), torch.empty((10,))),
            batch_size=32,
            data_loader_kwargs={'drop_last': drop_last}
        )
        assert data_loader.drop_last == drop_last

    def test_kwargs_and_batch_size(self):
        data_loader = _initialize_data_loader(
            dataset=torch.utils.data.TensorDataset(torch.empty((10, 3)), torch.empty((10,))),
            batch_size=32,
            data_loader_kwargs={'batch_size': 64}
        )
        assert data_loader.batch_size == 32

    @pytest.mark.parametrize('cls', [
        detector_cls
        for detector_cls in utils.all_classes('anomaly-detector', return_names=False)
        if issubclass(detector_cls, NeuralBaseDetector)
    ])
    def test_initialize_dataset(self, cls, multivariate_time_series):
        detector = cls(window_size=12)

        dataset = detector._initialize_dataset(multivariate_time_series)
        sample = dataset[0]
        assert len(sample) == 2  # Contains target and data

        data_loader = _initialize_data_loader(dataset, 8, None)
        data, targets = next(iter(data_loader))
        assert data.shape[0] == 8
        assert data.shape[1:] == sample[0].shape
        assert targets.shape[0] == 8
        assert targets.shape[1:] == sample[1].shape


class TestInitializeActivationFunction:

    @pytest.mark.parametrize('activation_name,expected_cls', [
        ('linear', torch.nn.Identity),
        ('relu', torch.nn.ReLU),
        ('sigmoid', torch.nn.Sigmoid),
        ('tanh', torch.nn.Tanh),
        ('leaky-relu', torch.nn.LeakyReLU),
    ])
    def test(self, activation_name, expected_cls):
        assert isinstance(_initialize_activation_function(activation_name), expected_cls)

    def test_invalid(self):
        with pytest.raises(ValueError):
            _initialize_activation_function('invalid')


class TestInitializeMLP:

    def test_base(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[8],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_hidden_layers(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_multiple_hidden_layers(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_batch_normalization(next(modules), 12)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 6, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_activation_multiple_values(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[8],
            activation_function=['sigmoid', 'tanh'],
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'sigmoid')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'tanh')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_activation_invalid_nb_values(self):
        with pytest.raises(ValueError):
            _initialize_mlp(
                input_dimension=16,
                output_dimension=4,
                hidden_layer_dimension=[8],
                activation_function=['relu', 'tanh', 'sigmoid'],
                batch_normalization=True,
                batch_normalization_first_layer=True,
                batch_normalization_last_layer=True,
                dropout_rate=0.2,
                dropout_first_layer=True,
                dropout_last_layer=True
            )

    def test_batch_normalization_multiple_values(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[8],
            activation_function='relu',
            batch_normalization=[False, True],
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_batch_normalization_invalid_nb_values(self):
        with pytest.raises(ValueError):
            _initialize_mlp(
                input_dimension=16,
                output_dimension=4,
                hidden_layer_dimension=[8],
                activation_function='relu',
                batch_normalization=[True, True, False],
                batch_normalization_first_layer=True,
                batch_normalization_last_layer=True,
                dropout_rate=0.2,
                dropout_first_layer=True,
                dropout_last_layer=True
            )

    def test_dropout_rate_multiple_values(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[8],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=[0.2, 0.3],
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.3)

        with pytest.raises(StopIteration):
            next(modules)

    def test_dropout_rate_invalid_nb_values(self):
        with pytest.raises(ValueError):
            _initialize_mlp(
                input_dimension=16,
                output_dimension=4,
                hidden_layer_dimension=[8],
                activation_function='relu',
                batch_normalization=True,
                batch_normalization_first_layer=True,
                batch_normalization_last_layer=True,
                dropout_rate=[0.2, 0.3, 0.4],
                dropout_first_layer=True,
                dropout_last_layer=True
            )

    def test_no_batch_normalization_first_layer(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=False,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 6, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_batch_normalization_last_layer(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=False,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_batch_normalization(next(modules), 12)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 6, 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_dropout_first_layer(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=False,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_batch_normalization(next(modules), 12)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 6, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        with pytest.raises(StopIteration):
            next(modules)

    def test_no_dropout_last_layer(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.2,
            dropout_first_layer=True,
            dropout_last_layer=False
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_batch_normalization(next(modules), 12)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 6, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_zero_dropout_all(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=0.0,
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_batch_normalization(next(modules), 12)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 6, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')

        with pytest.raises(StopIteration):
            next(modules)

    def test_zero_dropout_some(self):
        mlp = _initialize_mlp(
            input_dimension=16,
            output_dimension=4,
            hidden_layer_dimension=[12, 8, 6],
            activation_function='relu',
            batch_normalization=True,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=True,
            dropout_rate=[0.0, 0.2, 0.0, 0.3],
            dropout_first_layer=True,
            dropout_last_layer=True
        )
        modules = mlp.modules()

        assert is_sequential(next(modules))

        assert is_linear(next(modules), 16, 12)
        assert is_batch_normalization(next(modules), 12)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 12, 8)
        assert is_batch_normalization(next(modules), 8)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.2)

        assert is_linear(next(modules), 8, 6)
        assert is_batch_normalization(next(modules), 6)
        assert is_activation(next(modules), 'relu')

        assert is_linear(next(modules), 6, 4)
        assert is_batch_normalization(next(modules), 4)
        assert is_activation(next(modules), 'relu')
        assert is_dropout(next(modules), 0.3)

        with pytest.raises(StopIteration):
            next(modules)


class TestEdgeCasesFitting:  # Special cases of fitting that are not tested by default

    def test_no_validation(self, univariate_time_series):
        auto_encoder = AutoEncoder(window_size=12, validation_size=0.0, n_epochs=1)
        auto_encoder.fit(univariate_time_series)
        auto_encoder.decision_function(univariate_time_series)

    def test_data_loader_kwargs_given(self, univariate_time_series):
        auto_encoder = AutoEncoder(window_size=12, n_epochs=1, data_loader_kwargs={'batch_size': 16})
        auto_encoder.fit(univariate_time_series)
        auto_encoder.decision_function(univariate_time_series)

    def test_data_loader_kwargs_with_shuffle(self, univariate_time_series):
        auto_encoder = AutoEncoder(window_size=12, n_epochs=1, data_loader_kwargs={'shuffle': True})

        auto_encoder.fit(univariate_time_series)
        auto_encoder.decision_function(univariate_time_series)

        assert auto_encoder.data_loader_kwargs['shuffle']  # Check if shuffle is still true


class TestRandomState:

    @pytest.mark.parametrize('cls', [
        detector_cls
        for detector_cls in utils.all_classes('anomaly-detector', return_names=False)
        if issubclass(detector_cls, NeuralBaseDetector)
    ])
    def test(self, cls, univariate_time_series):
        y_pred1 = cls(window_size=12, random_state=0, n_epochs=1)\
            .fit(univariate_time_series)\
            .decision_function(univariate_time_series)
        y_pred2 = cls(window_size=12, random_state=0, n_epochs=1)\
            .fit(univariate_time_series)\
            .decision_function(univariate_time_series)
        assert np.array_equal(y_pred1, y_pred2)
