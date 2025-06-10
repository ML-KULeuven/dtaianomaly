from typing import List, Optional, Union

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.NeuralBaseDetector import (
    NeuralBaseDetector,
    _initialize_mlp,
)

# TODO would it make sense to allow for a completely custom architecture?


class AutoEncoder(NeuralBaseDetector):

    latent_space_dimension: int
    encoder_hidden_layer_dimension: List[int]
    decoder_hidden_layer_dimension: Optional[List[int]]
    encoder_activation_functions: Union[str, List[str]]
    decoder_activation_functions: Optional[Union[str, List[str]]]
    encoder_batch_normalization: Union[bool, List[bool]]
    decoder_batch_normalization: Optional[Union[bool, List[bool]]]
    encoder_dropout_rate: Union[float, List[float]]
    decoder_dropout_rate: Optional[Union[float, List[float]]]

    def __init__(
        self,
        window_size: Union[str, int],
        stride: int = 1,
        latent_space_dimension: int = 8,
        encoder_hidden_layer_dimension: List[int] = [64, 32],
        decoder_hidden_layer_dimension: Optional[List[int]] = None,
        encoder_activation_functions: Union[str, List[str]] = "relu",
        decoder_activation_functions: Optional[Union[str, List[str]]] = None,
        encoder_batch_normalization: Union[bool, List[bool]] = True,
        decoder_batch_normalization: Optional[Union[bool, List[bool]]] = None,
        encoder_dropout_rate: Union[float, List[float]] = 0.2,
        decoder_dropout_rate: Optional[Union[float, List[float]]] = None,
        scaling_method: str = None,
        batch_size: int = 32,
        data_loader_kwargs: Optional[dict] = None,
        validation_size: float = 0.1,
        optimizer: str = "Adam",
        learning_rate: float = 1e-3,
        optimizer_kwargs: Optional[dict] = None,
        n_epochs: int = 100,
        loss_function: torch.nn.Module = torch.nn.MSELoss(),
        device: str = "cpu",
    ):
        super().__init__(
            supervision=Supervision.SEMI_SUPERVISED,
            window_size=window_size,
            stride=stride,
            scaling_method=scaling_method,
            batch_size=batch_size,
            data_loader_kwargs=data_loader_kwargs,
            validation_size=validation_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
            n_epochs=n_epochs,
            loss_function=loss_function,
            device=device,
        )

        if not isinstance(latent_space_dimension, int) or isinstance(
            latent_space_dimension, bool
        ):
            raise TypeError("`latent_space_dimension` should be an integer")
        if latent_space_dimension < 1:
            raise ValueError("`latent_space_dimension` should be strictly positive")

        #########################################################################
        # ENCODER PARAMETER CHECKS
        #########################################################################

        if not utils.is_valid_list(encoder_hidden_layer_dimension, int):
            raise TypeError(
                "`encoder_hidden_layer_dimension` should be a list of integer"
            )
        if any(map(lambda x: x <= 0, encoder_hidden_layer_dimension)):
            raise ValueError(
                "All values in `encoder_hidden_layer_dimension` should be strictly positive"
            )
        nb_layers_encoder = len(encoder_hidden_layer_dimension) + 1

        if not (
            utils.is_valid_list(encoder_activation_functions, str)
            or isinstance(encoder_activation_functions, str)
        ):
            raise TypeError(
                "`encoder_activation_functions` should be a list of strings or a string"
            )
        # Values of encoder_activation_functions are checked in when initializing the architecture
        if (
            isinstance(encoder_activation_functions, list)
            and len(encoder_activation_functions) != nb_layers_encoder
        ):
            raise ValueError(
                f"Length `encoder_activation_functions` is {len(encoder_activation_functions)} but expected {nb_layers_encoder} (number of layers in decoder)"
            )

        if not (
            utils.is_valid_list(encoder_batch_normalization, bool)
            or isinstance(encoder_batch_normalization, bool)
        ):
            raise TypeError(
                "`encoder_batch_normalization` should be a list of bools or a bool"
            )
        if (
            isinstance(encoder_batch_normalization, list)
            and len(encoder_batch_normalization) != nb_layers_encoder
        ):
            raise ValueError(
                f"Length `encoder_activation_functions` is {len(encoder_batch_normalization)} but expected {nb_layers_encoder} (number of layers in decoder)"
            )

        if not (
            utils.is_valid_list(encoder_dropout_rate, float)
            or isinstance(encoder_dropout_rate, float)
        ):
            raise TypeError(
                "`encoder_dropout_rate` should be a list of floats or a float"
            )
        if isinstance(encoder_dropout_rate, list):
            if len(encoder_dropout_rate) != nb_layers_encoder:
                raise ValueError(
                    f"Length `encoder_dropout_rate` is {len(encoder_dropout_rate)} but expected {nb_layers_encoder} (number of layers in decoder)"
                )
            if not all(map(lambda x: 0.0 <= x < 1.0, encoder_dropout_rate)):
                raise ValueError(
                    f"All values in `encoder_dropout_rate` should be in interval [0, 1[."
                )
        else:
            if not 0.0 <= encoder_dropout_rate < 1.0:
                raise ValueError(
                    f"`encoder_dropout_rate` should be in interval [0, 1[."
                )

        #########################################################################
        # DECODER PARAMETER CHECKS
        #########################################################################

        if decoder_hidden_layer_dimension is not None:
            if not utils.is_valid_list(decoder_hidden_layer_dimension, int):
                raise TypeError(
                    "`decoder_hidden_layer_dimension` should be a list of integer"
                )
            if any(map(lambda x: x <= 0, decoder_hidden_layer_dimension)):
                raise ValueError(
                    "All values in `decoder_hidden_layer_dimension` should be strictly positive"
                )
            nb_layers_decoder = len(decoder_hidden_layer_dimension) + 1
        else:
            nb_layers_decoder = len(encoder_hidden_layer_dimension) + 1

        if decoder_activation_functions is not None:
            if not (
                utils.is_valid_list(decoder_activation_functions, str)
                or isinstance(decoder_activation_functions, str)
            ):
                raise TypeError(
                    "`decoder_activation_functions` should be a list of strings or a string"
                )
            # Values of encoder_activation_functions are checked in when initializing the architecture
            if (
                isinstance(decoder_activation_functions, list)
                and len(decoder_activation_functions) != nb_layers_decoder
            ):
                raise ValueError(
                    f"Length `decoder_activation_functions` is {len(decoder_activation_functions)} but expected {nb_layers_decoder} (number of layers in decoder)"
                )

        if decoder_batch_normalization is not None:
            if not (
                utils.is_valid_list(decoder_batch_normalization, bool)
                or isinstance(decoder_batch_normalization, bool)
            ):
                raise TypeError(
                    "`decoder_batch_normalization` should be a list of bools or a bool"
                )
            if (
                isinstance(decoder_batch_normalization, list)
                and len(decoder_batch_normalization) != nb_layers_decoder
            ):
                raise ValueError(
                    f"Length `decoder_batch_normalization` is {len(decoder_batch_normalization)} but expected {nb_layers_decoder} (number of layers in decoder)"
                )

        if decoder_dropout_rate is not None:
            if not (
                utils.is_valid_list(decoder_dropout_rate, float)
                or isinstance(decoder_dropout_rate, float)
            ):
                raise TypeError(
                    "`decoder_dropout_rate` should be a list of floats or a float"
                )
            if isinstance(decoder_dropout_rate, list):
                if len(decoder_dropout_rate) != nb_layers_decoder:
                    raise ValueError(
                        f"Length `decoder_dropout_rate` is {len(decoder_dropout_rate)} but expected {nb_layers_decoder} (number of layers in decoder)"
                    )
                if not all(map(lambda x: 0.0 <= x < 1.0, decoder_dropout_rate)):
                    raise ValueError(
                        f"All values in `decoder_dropout_rate` should be in interval [0, 1[."
                    )
            else:
                if not 0.0 <= decoder_dropout_rate < 1.0:
                    raise ValueError(
                        f"`decoder_dropout_rate` should be in interval [0, 1[."
                    )

        #########################################################################
        # INITIALIZATION OF THE PARAMETERS
        #########################################################################

        self.latent_space_dimension = latent_space_dimension
        self.encoder_hidden_layer_dimension = encoder_hidden_layer_dimension
        self.decoder_hidden_layer_dimension = decoder_hidden_layer_dimension
        self.encoder_activation_functions = encoder_activation_functions
        self.decoder_activation_functions = decoder_activation_functions
        self.encoder_batch_normalization = encoder_batch_normalization
        self.decoder_batch_normalization = decoder_batch_normalization
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate

        self._initialize_architecture(
            encoder_hidden_layer_dimension[0] * 2
        )  # Check if the architecture can be initialized

    def _initialize_dataset(self, X: np.ndarray) -> torch.utils.data.Dataset:
        # TODO probably a TensorDataSet
        raise NotImplementedError  # TODO

    def _initialize_architecture(self, input_size: int) -> torch.nn.Module:

        def _format_decoder_value(decoder_value, encoder_value):
            if decoder_value is not None:
                return decoder_value
            if isinstance(encoder_value, list):
                return encoder_value[::-1]
            return encoder_value

        return _AutoEncoderArchitecture(
            input_layer_dimension=input_size,
            latent_space_dimension=self.latent_space_dimension,
            encoder_hidden_layer_dimension=self.encoder_hidden_layer_dimension,
            decoder_hidden_layer_dimension=_format_decoder_value(
                decoder_value=self.decoder_hidden_layer_dimension,
                encoder_value=self.encoder_hidden_layer_dimension,
            ),
            encoder_activation_functions=self.encoder_activation_functions,
            decoder_activation_functions=_format_decoder_value(
                decoder_value=self.decoder_activation_functions,
                encoder_value=self.encoder_activation_functions,
            ),
            encoder_batch_normalization=self.encoder_batch_normalization,
            decoder_batch_normalization=_format_decoder_value(
                decoder_value=self.decoder_batch_normalization,
                encoder_value=self.encoder_batch_normalization,
            ),
            encoder_dropout_rate=self.encoder_dropout_rate,
            decoder_dropout_rate=_format_decoder_value(
                decoder_value=self.decoder_dropout_rate,
                encoder_value=self.encoder_dropout_rate,
            ),
        )

    def _compute_decision_scores(
        self, data_loader: torch.utils.data.DataLoader
    ) -> np.array:
        decision_scores = np.empty(len(data_loader))
        with torch.no_grad():
            for data, data_idx in data_loader:
                reconstruction = (
                    self.neural_network_(data.to(self.device).float()).cpu().numpy()
                )
                # TODO
                decision_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, reconstruction
                )
        return decision_scores


class _AutoEncoderArchitecture(torch.nn.Module):

    def __init__(
        self,
        input_layer_dimension: int,
        latent_space_dimension: int,
        encoder_hidden_layer_dimension: List[int],
        decoder_hidden_layer_dimension: List[int],
        encoder_activation_functions: Union[str, List[str]],
        decoder_activation_functions: Union[str, List[str]],
        encoder_batch_normalization: Union[bool, List[bool]],
        decoder_batch_normalization: Union[bool, List[bool]],
        encoder_dropout_rate: Union[float, List[float]],
        decoder_dropout_rate: Union[float, List[float]],
    ):
        super().__init__()

        # Inputs do not need to be checked here, as they are already checked in the
        # constructor of the auto encoder

        # Initialize the encoder
        self.encoder = _initialize_mlp(
            # initialize encoder and decoder as a sequential
            input_dimension=input_layer_dimension,
            output_dimension=latent_space_dimension,
            hidden_layer_dimension=encoder_hidden_layer_dimension,
            activation_function=encoder_activation_functions,
            batch_normalization=encoder_batch_normalization,
            batch_normalization_first_layer=False,
            batch_normalization_last_layer=True,
            dropout_rate=encoder_dropout_rate,
            dropout_first_layer=True,
            dropout_last_layer=True,
        )

        # Initialize the decoder
        self.decoder = _initialize_mlp(
            input_dimension=latent_space_dimension,
            output_dimension=input_layer_dimension,
            hidden_layer_dimension=decoder_hidden_layer_dimension,
            activation_function=decoder_activation_functions,
            batch_normalization=decoder_batch_normalization,
            batch_normalization_first_layer=True,
            batch_normalization_last_layer=False,
            dropout_rate=decoder_dropout_rate,
            dropout_first_layer=True,
            dropout_last_layer=False,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
