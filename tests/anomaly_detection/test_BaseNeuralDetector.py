import copy

import numpy as np
import pytest
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection import (
    AutoEncoder,
    BaseNeuralDetector,
    ConvolutionalNeuralNetwork,
    MultilayerPerceptron,
)
from dtaianomaly.anomaly_detection._BaseNeuralDetector import (
    ACTIVATION_FUNCTIONS,
    LOSSES,
    OPTIMIZERS,
)


def _neural_detectors_with(attribute):
    return [
        detector_cls
        for detector_cls in utils.all_classes("anomaly-detector", return_names=False)
        if issubclass(detector_cls, BaseNeuralDetector)
        and attribute in detector_cls.__dict__.get("attribute_validation", {})
    ]


class TestInitialize:

    def test_window_size_and_stride(self):
        # Quick and dirty tests as these things have generally been tested elsewhere
        MultilayerPerceptron(window_size="fft")
        MultilayerPerceptron(window_size="suss")
        MultilayerPerceptron(window_size=16)
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size=16.0)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size=-5)

        MultilayerPerceptron(window_size="fft", stride=1)
        MultilayerPerceptron(window_size="fft", stride=5)
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", stride=1.0)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", stride=-1)

    def test_standard_scaling_valid(self):
        detector = MultilayerPerceptron(window_size="fft", standard_scaling=False)
        assert detector.standard_scaling is False

    def test_standard_scaling_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", standard_scaling="invalid")

    def test_batch_size_valid(self):
        detector = MultilayerPerceptron(window_size="fft", batch_size=8)
        assert detector.batch_size == 8

    def test_batch_size_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", batch_size=8.0)

    def test_batch_size_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", batch_size=0)

    def test_optimizer_valid(self):
        detector = MultilayerPerceptron(window_size="fft", optimizer="sgd")
        assert detector.optimizer == "sgd"

    def test_optimizer_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", optimizer=10)

    def test_optimizer_invalid_callable(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(
                window_size="fft", optimizer=lambda x, y: torch.optim.Adagrad(x)
            )

    def test_optimizer_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", optimizer="invalid-optimizer")

    def test_learning_rate_valid(self):
        detector = MultilayerPerceptron(window_size="fft", learning_rate=0.01)
        assert detector.learning_rate == 0.01

    def test_learning_rate_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", learning_rate="0.001")

    def test_learning_rate_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", learning_rate=0.0)

    def test_compile_model_valid(self):
        detector = MultilayerPerceptron(window_size="fft", compile_model=True)
        assert detector.compile_model is True

    def test_compile_model_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", compile_model="invalid")

    def test_compile_mode_valid(self):
        detector = MultilayerPerceptron(window_size="fft", compile_mode="max-autotune")
        assert detector.compile_mode == "max-autotune"

    def test_compile_mode_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", compile_mode=5)

    def test_compile_mode_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", compile_mode="invalid")

    def test_n_epochs_valid(self):
        detector = MultilayerPerceptron(window_size="fft", n_epochs=5)
        assert detector.n_epochs == 5

    def test_n_epochs_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", n_epochs="100")

    def test_n_epochs_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", n_epochs=0)

    def test_loss_function_valid(self):
        detector = MultilayerPerceptron(window_size="fft", loss_function="huber")
        assert detector.loss_function == "huber"

    def test_loss_function_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", loss_function=10)

    def test_loss_function_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", loss_function="MSELoss")

    def test_device_cpu(self):
        detector = MultilayerPerceptron(window_size="fft", device="cpu")
        assert detector.device == "cpu"

    def test_device_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        detector = MultilayerPerceptron(window_size="fft", device="cuda")
        assert detector.device == "cuda"

    def test_device_cuda_not_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", device="cuda")

    def test_device_cuda_invalid_index(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", device="cuda:10000000")

    def test_device_invalid_type(self):
        with pytest.raises(TypeError):
            MultilayerPerceptron(window_size="fft", device=100)

    def test_device_invalid_value(self):
        with pytest.raises(ValueError):
            MultilayerPerceptron(window_size="fft", device="invalid-device")


class TestSharedSubclassParameters:

    @pytest.mark.parametrize("cls", _neural_detectors_with("dropout_rate"))
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.5])
    def test_dropout_rate_valid(self, cls, dropout_rate):
        detector = cls(window_size=16, dropout_rate=dropout_rate)
        assert detector.dropout_rate == dropout_rate

    @pytest.mark.parametrize("cls", _neural_detectors_with("dropout_rate"))
    def test_dropout_rate_invalid_type(self, cls):
        with pytest.raises(TypeError):
            cls(window_size=16, dropout_rate="0.1")

    @pytest.mark.parametrize("cls", _neural_detectors_with("dropout_rate"))
    @pytest.mark.parametrize("dropout_rate", [-0.1, 1.0])
    def test_dropout_rate_invalid_value(self, cls, dropout_rate):
        with pytest.raises(ValueError):
            cls(window_size=16, dropout_rate=dropout_rate)

    @pytest.mark.parametrize("cls", _neural_detectors_with("activation_function"))
    def test_activation_function_valid(self, cls):
        detector = cls(window_size=16, activation_function="tanh")
        assert detector.activation_function == "tanh"

    @pytest.mark.parametrize("cls", _neural_detectors_with("activation_function"))
    def test_activation_function_invalid_type(self, cls):
        with pytest.raises(TypeError):
            cls(window_size=16, activation_function=1)

    @pytest.mark.parametrize("cls", _neural_detectors_with("activation_function"))
    def test_activation_function_invalid_value(self, cls):
        with pytest.raises(ValueError):
            cls(window_size=16, activation_function="invalid-value")

    @pytest.mark.parametrize("cls", _neural_detectors_with("batch_normalization"))
    def test_batch_normalization_valid(self, cls):
        detector = cls(window_size=16, batch_normalization=False)
        assert detector.batch_normalization is False

    @pytest.mark.parametrize("cls", _neural_detectors_with("batch_normalization"))
    def test_batch_normalization_invalid_type(self, cls):
        with pytest.raises(TypeError):
            cls(window_size=16, batch_normalization="True")

    @pytest.mark.parametrize("cls", _neural_detectors_with("bias"))
    def test_bias_valid(self, cls):
        detector = cls(window_size=16, bias=False)
        assert detector.bias is False

    @pytest.mark.parametrize("cls", _neural_detectors_with("bias"))
    def test_bias_invalid_type(self, cls):
        with pytest.raises(TypeError):
            cls(window_size=16, bias="True")


class TestBuildDataLoader:

    @pytest.mark.parametrize(
        "cls",
        [
            detector_cls
            for detector_cls in utils.all_classes(
                "anomaly-detector", return_names=False
            )
            if issubclass(detector_cls, BaseNeuralDetector)
        ],
    )
    @pytest.mark.parametrize("univariate", [True, False])
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
        detector.optimizer_ = detector._build_optimizer(
            detector.neural_network_.parameters()
        )

        # Test if can be trained
        detector._train_batch(batch)

        # Test if can be evaluated
        detector._evaluate_batch(batch)

    @pytest.mark.parametrize("drop_last", [True, False])
    def test_kwargs(self, drop_last, univariate_time_series):
        detector = MultilayerPerceptron(
            window_size=1, data_loader_kwargs={"drop_last": drop_last}
        )
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(
            detector._build_dataset(univariate_time_series)
        )
        assert data_loader.drop_last == drop_last

    def test_kwargs_and_batch_size(self, univariate_time_series):
        detector = MultilayerPerceptron(
            window_size=1, batch_size=32, data_loader_kwargs={"batch_size": 64}
        )
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(
            detector._build_dataset(univariate_time_series)
        )
        assert data_loader.batch_size == 32
        assert detector.data_loader_kwargs["batch_size"] == 64

    @pytest.mark.parametrize("forced_shuffle", [True, False])
    def test_shuffle_no_given_kwargs(self, univariate_time_series, forced_shuffle):
        detector = MultilayerPerceptron(window_size=1)
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(
            detector._build_dataset(univariate_time_series), shuffle=forced_shuffle
        )
        is_shuffled = isinstance(data_loader.sampler, torch.utils.data.RandomSampler)
        assert is_shuffled == forced_shuffle

    @pytest.mark.parametrize("forced_shuffle", [True, False])
    @pytest.mark.parametrize("given_shuffle", [True, False])
    def test_shuffle_given_kwargs(
        self, univariate_time_series, forced_shuffle, given_shuffle
    ):
        detector = MultilayerPerceptron(
            window_size=1, data_loader_kwargs={"shuffle": given_shuffle}
        )
        detector.window_size_ = detector.window_size
        data_loader = detector._build_data_loader(
            detector._build_dataset(univariate_time_series), shuffle=forced_shuffle
        )
        is_shuffled = isinstance(data_loader.sampler, torch.utils.data.RandomSampler)
        assert is_shuffled == forced_shuffle
        assert detector.data_loader_kwargs["shuffle"] == given_shuffle


class TestBuildDataLoaderDropLast:

    @staticmethod
    def _data_loader(n_samples, batch_size=32, train=True, data_loader_kwargs=None):
        detector = MultilayerPerceptron(
            window_size=1, batch_size=batch_size, data_loader_kwargs=data_loader_kwargs
        )
        dataset = torch.utils.data.TensorDataset(torch.zeros(n_samples, 1))
        return detector._build_data_loader(dataset, train=train)

    @pytest.mark.parametrize(
        "n_samples,batch_size,expected",
        [
            (33, 32, True),
            (65, 32, True),
            (34, 32, False),
            (64, 32, False),
            (32, 32, False),
            (1, 32, False),
            (1, 1, False),
            (5, 1, False),
        ],
    )
    def test_train(self, n_samples, batch_size, expected):
        data_loader = self._data_loader(n_samples, batch_size)
        assert data_loader.drop_last == expected

    @pytest.mark.parametrize("n_samples", [33, 34, 1])
    def test_evaluation_never_drops(self, n_samples):
        assert not self._data_loader(n_samples, train=False).drop_last

    @pytest.mark.parametrize("n_samples", [33, 34])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_train_respects_user_kwargs(self, n_samples, drop_last):
        data_loader = self._data_loader(
            n_samples, data_loader_kwargs={"drop_last": drop_last}
        )
        assert data_loader.drop_last == drop_last


class TestFitBatchNormalization:

    @pytest.mark.parametrize(
        "cls", [AutoEncoder, MultilayerPerceptron, ConvolutionalNeuralNetwork]
    )
    def test_last_batch_of_single_window(self, cls):
        X = np.sin(np.arange(300) / 5)
        y_pred = cls(window_size=12, n_epochs=1, seed=0).fit(X).decision_function(X)
        assert y_pred.shape == (300,)

    @pytest.mark.parametrize(
        "cls,n_samples",
        [
            (AutoEncoder, 12),
            (MultilayerPerceptron, 13),
            (ConvolutionalNeuralNetwork, 13),
        ],
    )
    def test_single_window(self, cls, n_samples):
        X = np.sin(np.arange(n_samples) / 5)
        with pytest.raises(ValueError, match="At least 2 windows are required"):
            cls(window_size=12, n_epochs=1).fit(X)

    def test_single_window_without_batch_normalization(self):
        X = np.sin(np.arange(13) / 5)
        detector = MultilayerPerceptron(
            window_size=12, n_epochs=1, batch_normalization=False
        )
        detector.fit(X)
        assert detector.decision_function(X).shape == (13,)


class TestTrain:

    def test_restores_best_epoch_weights(self, monkeypatch):
        losses = [3.0, 1.0, 2.0, 5.0]
        snapshots = []
        detector = MultilayerPerceptron(
            window_size=8, n_epochs=len(losses), batch_size=64, seed=0
        )

        def _train_batch(batch):
            with torch.no_grad():
                for parameter in detector.neural_network_.parameters():
                    parameter.add_(1.0)
            snapshots.append(copy.deepcopy(detector.neural_network_.state_dict()))
            return losses[len(snapshots) - 1]

        monkeypatch.setattr(detector, "_train_batch", _train_batch)
        detector.fit(np.sin(np.arange(40) / 5))

        state_dict = detector.neural_network_.state_dict()
        assert all(torch.equal(state_dict[k], snapshots[1][k]) for k in state_dict)
        assert not all(torch.equal(state_dict[k], snapshots[-1][k]) for k in state_dict)


class TestBuildActivationFunction:

    @pytest.mark.parametrize("activation_function", ACTIVATION_FUNCTIONS)
    def test(self, activation_function):  # All defined functions work
        BaseNeuralDetector._build_activation_function(activation_function)

    @pytest.mark.parametrize(
        "activation_function,cls",
        [
            ("linear", torch.nn.Identity),
            ("relu", torch.nn.ReLU),
            ("sigmoid", torch.nn.Sigmoid),
            ("tanh", torch.nn.Tanh),
        ],
    )
    def test_valid_return(self, activation_function, cls):
        assert isinstance(
            BaseNeuralDetector._build_activation_function(activation_function), cls
        )

    def test_invalid(self):
        with pytest.raises(ValueError):
            BaseNeuralDetector._build_activation_function("invalid")


class TestBuildOptimizer:

    @pytest.mark.parametrize("optimizer", OPTIMIZERS)
    def test(self, optimizer):  # All defined functions work
        detector = MultilayerPerceptron(window_size=1, optimizer=optimizer)
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        detector._build_optimizer(architecture.parameters())

    @pytest.mark.parametrize(
        "optimizer,cls",
        [
            ("adam", torch.optim.Adam),
            ("sgd", torch.optim.SGD),
        ],
    )
    def test_valid_return(self, optimizer, cls):
        detector = MultilayerPerceptron(window_size=1, optimizer=optimizer)
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        assert isinstance(detector._build_optimizer(architecture.parameters()), cls)

    @pytest.mark.parametrize("learning_rate", [0.01, 0.00005])
    def test_learning_rate(self, learning_rate):
        detector = MultilayerPerceptron(window_size=1, learning_rate=learning_rate)
        detector.window_size_ = detector.window_size
        architecture = detector._build_architecture(100)
        optimizer = detector._build_optimizer(architecture.parameters())
        assert optimizer.param_groups[0]["lr"] == learning_rate


class TestBuildLossFunction:

    @pytest.mark.parametrize("loss", LOSSES)
    def test(self, loss):  # All defined functions work
        MultilayerPerceptron(16, loss_function=loss)

    @pytest.mark.parametrize(
        "loss,cls",
        [
            ("mse", torch.nn.MSELoss),
            ("l1", torch.nn.L1Loss),
            ("huber", torch.nn.SmoothL1Loss),
        ],
    )
    def test_valid_return(self, loss, cls):
        assert isinstance(
            MultilayerPerceptron(16, loss_function=loss)._build_loss_function(), cls
        )


@pytest.mark.slow
class TestCompile:

    @pytest.mark.parametrize("compile_mode", ["default"])
    def test(self, compile_mode, univariate_time_series):
        detector = MultilayerPerceptron(
            window_size=12, compile_model=True, compile_mode=compile_mode
        )
        detector.fit(univariate_time_series)
        detector.decision_function(univariate_time_series)


@pytest.mark.slow
class TestRandomState:

    @pytest.mark.parametrize(
        "cls",
        [
            detector_cls
            for detector_cls in utils.all_classes(
                "anomaly-detector", return_names=False
            )
            if issubclass(detector_cls, BaseNeuralDetector)
        ],
    )
    def test(self, cls, univariate_time_series):
        # Avoid a last batch of size 1, which fails with batch normalization
        X = univariate_time_series[:320]
        y_pred1 = cls(window_size=12, seed=0, n_epochs=1).fit(X).decision_function(X)
        y_pred2 = cls(window_size=12, seed=0, n_epochs=1).fit(X).decision_function(X)
        assert np.allclose(y_pred1, y_pred2)
