import torch

from dtaianomaly.anomaly_detection.BaseNeuralDetector import BaseNeuralDetector


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


def is_normalization(module, normalized_shape):
    if isinstance(module, torch.nn.LayerNorm):
        return module.normalized_shape == normalized_shape
    return False


def is_activation(module, activation):
    return isinstance(module, BaseNeuralDetector._ACTIVATION_FUNCTIONS[activation])


def is_dropout(module, p):
    if isinstance(module, torch.nn.Dropout):
        return module.p == p
    return False


def is_flatten(module):
    return isinstance(module, torch.nn.Flatten)


def is_un_flatten(module):
    if isinstance(module, torch.nn.Unflatten):
        return True
    return False


def is_lstm(module, input_size, hidden_size, num_layers, bias, dropout):
    if isinstance(module, torch.nn.LSTM):
        return (
            module.input_size == input_size
            and module.hidden_size == hidden_size
            and module.num_layers == num_layers
            and module.bias == bias
            and module.dropout == dropout
        )
    return False


def is_conv1d(module, in_channels, out_channels, kernel_size, padding):
    if isinstance(module, torch.nn.Conv1d):
        return (
            module.in_channels == in_channels
            and module.out_channels == out_channels
            and module.kernel_size == kernel_size
            and module.padding == padding
        )
    return False


def is_avg_pooling(module):
    if isinstance(module, torch.nn.AvgPool1d):
        return True
    return False


def is_transformer_encoder(module, num_layers, enable_nested_tensor):
    if isinstance(module, torch.nn.TransformerEncoder):
        return (
            module.num_layers == num_layers
            and module.enable_nested_tensor == enable_nested_tensor
        )
    return False


def is_transformer_encoder_layer(module):
    return isinstance(module, torch.nn.TransformerEncoderLayer)


def is_multihead_attention(module, num_heads):
    if isinstance(module, torch.nn.MultiheadAttention):
        return module.num_heads == num_heads
    return False


def is_non_dynamically_quantizable_linear(module, in_features, out_features):
    if isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
        return module.in_features == in_features and module.out_features == out_features
    return False
