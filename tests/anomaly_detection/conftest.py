
import torch
from dtaianomaly.anomaly_detection.NeuralBaseDetector import _initialize_activation_function


def is_sequential(module):
    return isinstance(module, torch.nn.Sequential)


def is_linear(module, in_features, out_features):
    if isinstance(module, torch.nn.Linear):
        return module.in_features == in_features and module.out_features == out_features
    return False


def is_batch_normalization(module, num_features):
    if isinstance(module, torch.nn.BatchNorm1d):
        return module.num_features == num_features
    return False


def is_activation(module, activation):
    return isinstance(module, type(_initialize_activation_function(activation)))


def is_dropout(module, p):
    if isinstance(module, torch.nn.Dropout):
        return module.p == p
    return False
