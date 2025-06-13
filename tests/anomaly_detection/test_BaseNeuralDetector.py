
import pytest
import torch
import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection import MultilayerPerceptron, BaseNeuralDetector, ForecastDataset, ReconstructionDataset, AutoEncoder

_VALID_ERROR_METRICS = ["mean-absolute-error", "mean-squared-error"]


class TestNeuralBaseDetector:

    def test_str(self):
        assert str(MultilayerPerceptron(window_size=100)) == "MultilayerPerceptron(window_size=100)"
        assert str(MultilayerPerceptron(window_size=100, learning_rate=0.01)) == "MultilayerPerceptron(window_size=100,learning_rate=0.01)"
        assert str(MultilayerPerceptron(window_size=100, loss_function=torch.nn.L1Loss())) == "MultilayerPerceptron(window_size=100,loss_function=L1Loss())"
        assert str(MultilayerPerceptron(window_size=100, data_loader_kwargs={'shuffle': True})) == "MultilayerPerceptron(window_size=100,data_loader_kwargs={'shuffle': True})"


class TestInitialize:

    def test_window_size_and_stride(self):
        # Quick and dirty tests as these things have generally been tested elsewhere
        MultilayerPerceptron(window_size='fft')
        MultilayerPerceptron(window_size='suss')
        MultilayerPerceptron(window_size=16)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=16.0)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=-5)

        MultilayerPerceptron(window_size='fft', stride=1)
        MultilayerPerceptron(window_size='fft', stride=5)
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', stride=1.0)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', stride=-1)

    @pytest.mark.parametrize('standard_scaling', [True, False])
    def test_standard_scaling_valid(self, standard_scaling):
        detector = MultilayerPerceptron(window_size='fft', standard_scaling=standard_scaling)
        assert detector.standard_scaling == standard_scaling

    @pytest.mark.parametrize('standard_scaling', [5, 1.0, 'invalid'])
    def test_standard_scaling_invalid_type(self, standard_scaling):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', standard_scaling=standard_scaling)

    @pytest.mark.parametrize('batch_size', [8, 16, 32])
    def test_batch_size_valid(self, batch_size):
        detector = MultilayerPerceptron(window_size='fft', batch_size=batch_size)
        assert detector.batch_size == batch_size

    @pytest.mark.parametrize('batch_size', ['8', 8.0])
    def test_batch_size_invalid_type(self, batch_size):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', batch_size=batch_size)

    @pytest.mark.parametrize('batch_size', [0, -8])
    def test_batch_size_invalid_value(self, batch_size):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', batch_size=batch_size)

    def test_data_loader_kwargs_valid(self):
        detector = MultilayerPerceptron(window_size='fft', data_loader_kwargs={'shuffle': True})
        assert detector.data_loader_kwargs == {'shuffle': True}

    def test_data_loader_kwargs_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', data_loader_kwargs={'invalid-kwarg': True})
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', data_loader_kwargs='invalid-kwarg')

    @pytest.mark.parametrize('optimizer', BaseNeuralDetector._OPTIMIZERS.keys())
    def test_optimizer_valid(self, optimizer):
        detector = MultilayerPerceptron(window_size='fft', optimizer=optimizer)
        assert detector.optimizer == optimizer

    def test_optimizer_valid_callable(self):
        MultilayerPerceptron(window_size='fft', optimizer=lambda x: torch.optim.Adagrad(x))

    @pytest.mark.parametrize('optimizer', [10, False])
    def test_optimizer_invalid_type(self, optimizer):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', optimizer=optimizer)

    def test_optimizer_invalid_callable(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', optimizer=lambda x, y: torch.optim.Adagrad(x))

    @pytest.mark.parametrize('optimizer', ['invalid-optimizer'])
    def test_optimizer_invalid_value(self, optimizer):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', optimizer=optimizer)

    @pytest.mark.parametrize('learning_rate', [0.01, 1.0, 10])
    def test_learning_rate_valid(self, learning_rate):
        detector = MultilayerPerceptron(window_size='fft', learning_rate=learning_rate)
        assert detector.learning_rate == learning_rate

    @pytest.mark.parametrize('learning_rate', ['0.001', False])
    def test_learning_rate_invalid_type(self, learning_rate):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', learning_rate=learning_rate)

    @pytest.mark.parametrize('learning_rate', [-1, 0])
    def test_learning_rate_invalid_value(self, learning_rate):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', learning_rate=learning_rate)

    @pytest.mark.parametrize('compile_model', [True, False])
    def test_compile_model_valid(self, compile_model):
        detector = MultilayerPerceptron(window_size='fft', compile_model=compile_model)
        assert detector.compile_model == compile_model

    @pytest.mark.parametrize('compile_model', [5, 1.0, 'invalid'])
    def test_compile_model_invalid_type(self, compile_model):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', compile_model=compile_model)

    @pytest.mark.parametrize('compile_mode', [
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs"
    ])
    def test_compile_mode_valid(self, compile_mode):
        detector = MultilayerPerceptron(window_size='fft', compile_mode=compile_mode)
        assert detector.compile_mode == compile_mode

    @pytest.mark.parametrize('compile_mode', [5, 1.0, True])
    def test_compile_mode_invalid_type(self, compile_mode):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', compile_mode=compile_mode)

    @pytest.mark.parametrize('compile_mode', ['invalid'])
    def test_compile_mode_invalid_value(self, compile_mode):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', compile_mode=compile_mode)

    @pytest.mark.parametrize('n_epochs', [5, 100])
    def test_n_epochs_valid(self, n_epochs):
        detector = MultilayerPerceptron(window_size='fft', n_epochs=n_epochs)
        assert detector.n_epochs == n_epochs

    @pytest.mark.parametrize('n_epochs', [True, '100'])
    def test_n_epochs_invalid_type(self, n_epochs):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', n_epochs=n_epochs)

    @pytest.mark.parametrize('n_epochs', [-1, 0])
    def test_n_epochs_invalid_value(self, n_epochs):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', n_epochs=n_epochs)

    @pytest.mark.parametrize('loss_function', [torch.nn.MSELoss(), torch.nn.L1Loss()])
    def test_loss_function_valid(self, loss_function):
        detector = MultilayerPerceptron(window_size='fft', loss_function=loss_function)
        assert detector.loss_function == loss_function

    @pytest.mark.parametrize('loss_function', ['torch.nn.MSELoss()', 'MSELoss', 10])
    def test_loss_function_invalid_type(self, loss_function):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', loss_function=loss_function)

    def test_device_cpu(self):
        detector = MultilayerPerceptron(window_size='fft', device='cpu')
        assert detector.device == 'cpu'

    def test_device_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        detector = MultilayerPerceptron(window_size='fft', device='cuda')
        assert detector.device == 'cuda'

    def test_device_cuda_not_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', device='cuda')

    def test_device_cuda_invalid_index(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', device='cuda:10000000')

    @pytest.mark.parametrize('device', [True, 100])
    def test_device_invalid_type(self, device):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size='fft', device=device)

    @pytest.mark.parametrize('device', ['invalid-device'])
    def test_device_invalid_value(self, device):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size='fft', device=device)


class TestBuildDataLoader:

    @pytest.mark.parametrize('cls', [
        detector_cls
        for detector_cls in utils.all_classes('anomaly-detector', return_names=False)
        if issubclass(detector_cls, BaseNeuralDetector)
    ])
    @pytest.mark.parametrize('univariate', [True, False])
    def test(self, cls, univariate, univariate_time_series, multivariate_time_series):
        detector = cls(window_size=8)
        X = univariate_time_series if univariate else multivariate_time_series

        # Build the data loader
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(detector._build_dataset(X))
        batch = next(iter(data_loader))
        print(batch[0].shape)
        print(len(batch))
        assert batch[0].shape[0] == detector.batch_size

        detector.window_size_ = detector.window_size
        detector.neural_network_ = detector._build_architecture(utils.get_dimension(X))
        detector.optimizer_ = detector._build_optimizer(detector.neural_network_.parameters())

        # Test if can be trained
        detector._train_batch(batch)

        # Test if can be evaluated
        detector._evaluate_batch(batch)

    @pytest.mark.parametrize('drop_last', [True, False])
    def test_kwargs(self, drop_last, univariate_time_series):
        detector = MultilayerPerceptron(window_size=1, data_loader_kwargs={'drop_last': drop_last})
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(detector._build_dataset(univariate_time_series))
        assert data_loader.drop_last == drop_last

    def test_kwargs_and_batch_size(self, univariate_time_series):
        detector = MultilayerPerceptron(window_size=1, batch_size=32, data_loader_kwargs={'batch_size': 64})
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(detector._build_dataset(univariate_time_series))
        assert data_loader.batch_size == 32
        assert detector.data_loader_kwargs['batch_size'] == 64

    @pytest.mark.parametrize('forced_shuffle', [True, False])
    def test_shuffle_no_given_kwargs(self, univariate_time_series, forced_shuffle):
        detector = MultilayerPerceptron(window_size=1)
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(detector._build_dataset(univariate_time_series), shuffle=forced_shuffle)
        is_shuffled = isinstance(data_loader.sampler, torch.utils.data.RandomSampler)
        assert is_shuffled == forced_shuffle

    @pytest.mark.parametrize('forced_shuffle', [True, False])
    @pytest.mark.parametrize('given_shuffle', [True, False])
    def test_shuffle_given_kwargs(self, univariate_time_series, forced_shuffle, given_shuffle):
        detector = MultilayerPerceptron(window_size=1, data_loader_kwargs={'shuffle': given_shuffle})
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(detector._build_dataset(univariate_time_series), shuffle=forced_shuffle)
        is_shuffled = isinstance(data_loader.sampler, torch.utils.data.RandomSampler)
        assert is_shuffled == forced_shuffle
        assert detector.data_loader_kwargs['shuffle'] == given_shuffle


class TestBuildActivationFunction:

    @pytest.mark.parametrize('activation_function,cls', BaseNeuralDetector._ACTIVATION_FUNCTIONS.items())
    def test(self, activation_function, cls):
        assert isinstance(BaseNeuralDetector._build_activation_function(activation_function), cls)

    def test_invalid(self):
        with pytest.raises(ValueError):
            BaseNeuralDetector._build_activation_function('invalid')


class TestBuildOptimizer:

    @pytest.mark.parametrize('optimizer,cls', BaseNeuralDetector._OPTIMIZERS.items())
    def test(self, optimizer, cls):
        detector = MultilayerPerceptron(window_size=1, optimizer=optimizer)
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        assert isinstance(detector._build_optimizer(architecture.parameters()), cls)

    def test_invalid_optimizer(self):
        detector = MultilayerPerceptron(window_size=1, optimizer='adam')
        detector.window_size_ = detector.window_size
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        detector.optimizer = 'invalid'
        with pytest.raises(ValueError):
            detector._build_optimizer(architecture.parameters())

    @pytest.mark.parametrize('learning_rate', [0.01, 0.00005])
    def test_learning_rate(self, learning_rate):
        detector = MultilayerPerceptron(window_size=1, learning_rate=learning_rate)
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        optimizer = detector._build_optimizer(architecture.parameters())
        assert optimizer.param_groups[0]['lr'] == learning_rate

    def test_learning_rate_with_callable_optimizer(self):
        detector = MultilayerPerceptron(window_size=1, learning_rate=10)
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        optimizer = detector._build_optimizer(architecture.parameters())
        assert optimizer.param_groups[0]['lr'] == 10

        detector = MultilayerPerceptron(window_size=1, learning_rate=10, optimizer=lambda x: torch.optim.Adam(x, lr=0.1))
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        optimizer = detector._build_optimizer(architecture.parameters())
        assert optimizer.param_groups[0]['lr'] == 0.1


class TestCompile:

    @pytest.mark.parametrize('compile_mode', ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    def test(self, compile_mode, univariate_time_series):
        detector = MultilayerPerceptron(window_size=12, compile_model=True, compile_mode=compile_mode)
        detector.fit(univariate_time_series)
        detector.decision_function(univariate_time_series)


class TestRandomState:

    @pytest.mark.parametrize('cls', [
        detector_cls
        for detector_cls in utils.all_classes('anomaly-detector', return_names=False)
        if issubclass(detector_cls, BaseNeuralDetector)
    ])
    def test(self, cls, univariate_time_series):
        y_pred1 = cls(window_size=12, seed=0, n_epochs=1)\
            .fit(univariate_time_series)\
            .decision_function(univariate_time_series)
        y_pred2 = cls(window_size=12, seed=0, n_epochs=1)\
            .fit(univariate_time_series)\
            .decision_function(univariate_time_series)
        assert np.allclose(y_pred1, y_pred2)
