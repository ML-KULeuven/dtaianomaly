import sys
from typing import Literal

import numpy as np
import torch

from dtaianomaly import utils
from dtaianomaly.anomaly_detection._BaseDetector import BaseDetector, Supervision
from dtaianomaly.type_validation import (
    BoolAttribute,
    FloatAttribute,
    IntegerAttribute,
    LiteralAttribute,
    WindowSizeAttribute,
)
from dtaianomaly.windowing import WINDOW_SIZE_TYPE, compute_window_size

__all__ = ["MOMENT", "MODEL_SIZES"]

MODEL_SIZE_TYPE = Literal["small", "base", "large"]
MODEL_SIZES = ["small", "base", "large"]


class MOMENT(BaseDetector):
    """
    Detect anomalies in time series using MOMENT :cite:`goswami2024moment`.

    MOMENT is pre-trained time series foundation model. MOMENT is trained to
    reconstruct masked patches in the time series, thereby learning internal
    time series representations. The architecture of MOMENT consists of a
    encoder-only transformer, and a lightweight reconstruction head.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:func:`~dtaianomaly.anomaly_detection.windowing_utils.compute_window_size`.
    model_size: {'small', 'base', 'large'}, default='small'
        The MOMENT-model to use.
    batch_size: int, default=16
        The number of windows to feed simultaneously to Chronos, within a batch.
    do_fine_tuning: bool, default=False
        Whether to fine tune the model during fitting. If False, then the model will
        perform zero-shot forecasting.
    learning_rate: float, default=1e-4
        The learning rate to use for fine-tuning MOMENT.
    nb_epochs: int, default=1
        The number of epochs to finetune MOMENT.
    device: str, default='cpu'
        The device to use.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    moment_ : momentfm.MOMENTPipeline
        The MOMENT model

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import MOMENT  # doctest: +SKIP
    >>> from dtaianomaly.data import demonstration_time_series  # doctest: +SKIP
    >>> x, y = demonstration_time_series()  # doctest: +SKIP
    >>> moment = MOMENT(10).fit(x)  # doctest: +SKIP
    >>> moment.decision_function(x)  # doctest: +SKIP
    array([0.00027719, 0.00027719, 0.00027719, ..., 0.00058781, 0.02628242,
           0.00010728]...)

    Warnings
    --------
    MOMENT only works for Python 3.11. Additionally, its requirements are very strict, and
    must be installed seperately from dtaianomaly. This can be done via ``pip install momentfm``.


    Notes
    -----
    MOMENTAnomalyDetector only handles univariate time series.
    """

    window_size: WINDOW_SIZE_TYPE
    model_size: MODEL_SIZE_TYPE
    batch_size: int
    do_fine_tuning: bool
    learning_rate: float
    nb_epochs: int
    device: str

    window_size_: int
    moment_: any

    attribute_validation = {
        "window_size": WindowSizeAttribute(),
        "model_size": LiteralAttribute(MODEL_SIZES),
        "batch_size": IntegerAttribute(1),
        "do_fine_tuning": BoolAttribute(),
        "learning_rate": FloatAttribute(0.0, inclusive_minimum=False),
        "nb_epochs": IntegerAttribute(1),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        model_size: MODEL_SIZE_TYPE = "small",
        batch_size: int = 16,
        do_fine_tuning: bool = False,
        learning_rate: float = 1e-4,
        nb_epochs: int = 1,
        device: str = "cpu",
    ):
        if sys.version_info[:2] != (3, 11):
            raise EnvironmentError(
                f"MOMENT requires Python 3.11! Current version is {sys.version.split()[0]}"
            )

        try:
            import momentfm
        except ImportError:
            raise Exception(
                "Module 'momentfm' is not available, make sure you install it before using MOMENT!"
            )
        super().__init__(Supervision.UNSUPERVISED)
        self.window_size = window_size
        self.model_size = model_size
        self.batch_size = batch_size
        self.do_fine_tuning = do_fine_tuning
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.device = device

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Compute the window size
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)

        device = torch.device(self.device)

        from momentfm import MOMENTPipeline

        self.moment_ = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{self.model_size}",
            model_kwargs={
                "task_name": "reconstruction",  # Set task to reconstruction
                "freeze_encoder": True,  # Only train the head, not the encoder
                "enable_grad_checkpoint": False,  # Disable gradient checkpointing
            },
        )
        self.moment_.init()
        self.moment_ = self.moment_.to(device).float()

        # Fine-tune the reconstruction head if enabled
        if self.do_fine_tuning:
            # Set the moment in train mode
            self.moment_.train()
            optimizer = torch.optim.Adam(
                self.moment_.head.parameters(), lr=self.learning_rate
            )  # Only optimize the head
            criterion = torch.nn.MSELoss()

            batches_and_masks = self._create_batches_and_mask(X)
            for epoch in range(self.nb_epochs):

                # Process windows in batches
                for batch, masks in batches_and_masks:
                    # Forward pass and optimization
                    optimizer.zero_grad()
                    out = self.moment_(x_enc=batch, input_mask=masks)
                    loss = criterion(out.reconstruction.squeeze(1), batch.squeeze(1))
                    loss.backward()
                    optimizer.step()

            # Switch back to evaluation mode
            self.moment_.eval()

    def _decision_function(self, X: np.ndarray) -> np.array:

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Predict the anomaly scores for each batch
        anomaly_criterion = torch.nn.MSELoss(reduction="none")
        decision_scores = np.empty(shape=X.shape[0])
        decision_scores = np.full_like(decision_scores, np.nan)
        with torch.no_grad():

            for i, (batch, masks) in enumerate(self._create_batches_and_mask(X)):
                output = self.moment_(x_enc=batch, input_mask=masks)
                error = (
                    torch.mean(anomaly_criterion(batch, output.reconstruction), dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()
                )
                start = i * self.batch_size + (self.window_size_ // 2)
                decision_scores[start : start + error.shape[0]] = error

        # Padding
        decision_scores[: (self.window_size_ // 2)] = decision_scores[
            (self.window_size_ // 2)
        ]
        decision_scores[-(self.window_size_ // 2) :] = decision_scores[
            -(self.window_size_ // 2)
        ]

        return decision_scores

    def _create_batches_and_mask(
        self, X: np.ndarray
    ) -> list[(torch.tensor, torch.tensor)]:
        X = X.squeeze()

        nb_windows = X.shape[0] - self.window_size_ + 1
        nb_complete_batches = nb_windows // self.batch_size
        nb_remaining_windows = nb_windows - nb_complete_batches * self.batch_size

        batches = []
        for i in range(nb_complete_batches):
            batches.append(
                np.array(
                    [
                        X[
                            (i * self.batch_size)
                            + t : (i * self.batch_size)
                            + t
                            + self.window_size_
                        ].T
                        for t in range(self.batch_size)
                    ]
                )
            )

        if nb_remaining_windows > 0:
            batches.append(
                np.array(
                    [
                        X[
                            (nb_complete_batches * self.batch_size)
                            + t : (nb_complete_batches * self.batch_size)
                            + t
                            + self.window_size_
                        ].T
                        for t in range(nb_remaining_windows)
                    ]
                )
            )

        batches_and_masks = []
        for batch in batches:
            batch = torch.tensor(batch)
            masks = torch.ones_like(batch, dtype=torch.float64).to(self.device)
            batch = batch.unsqueeze(1).to(self.device).float()
            batches_and_masks.append((batch, masks))

        return batches_and_masks
