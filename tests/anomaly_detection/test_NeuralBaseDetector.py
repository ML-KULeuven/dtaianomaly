
import pytest
import torch
from dtaianomaly.anomaly_detection import AutoEncoder
from dtaianomaly.anomaly_detection.NeuralBaseDetector import _initialize_optimizer, _initialize_activation_function, _initialize_data_loader, _initialize_mlp


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

    def test(self):
        assert 0
