import numpy as np

from dtaianomaly.evaluation._common import make_intervals
from dtaianomaly.evaluation.metrics import ProbaMetric


class UCRScore(ProbaMetric):
    """
    Compute the UCR-Score to evaluate the predicted anomaly scores :cite:`wu2023current`.

    The UCR-score is a binary metric for a time series with a *single* anomalous event
    starting at time :math:`t_s` until time :math:`t_e` of length :math:`w = t_e - t_s + 1`.
    Denote :math:`t^*` as the time step with  the highest anomaly score. We define tolerance
    :math:`\\xi` to reduce bias towards shorter anomalies :cite:`rewicki2023is`. The UCR Score
    equals 1 if :math:`t^*` is near the anomalous event, and 0 otherwise. Formally:

    .. math::
        UCR_{\\text{score}, \\xi} =
        \\begin{cases}
            1 & t_s - \\max(w, \\xi) \\leq t^* \\leq t_e + \\max(w, \\xi) \\\\
            0 & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    tolerance: int, default=None
        The minimum tolerance around the ground truth anomalous event to avoid
        bias towards short anomalies. If None, no tolerance is included.
    """

    tolerance: int | None

    def __init__(self, tolerance: int = None):
        if tolerance is not None:
            if not isinstance(tolerance, int) or isinstance(tolerance, bool):
                raise TypeError("`tolerance` should be an integer")
            if tolerance < 1:
                raise ValueError("`tolerance`  should be at least 1!")
        self.tolerance = tolerance

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:

        # Retrieve the ground truth anomalous interval
        starts, ends = make_intervals(y_true)
        if len(starts) != 1:
            raise ValueError(
                "The 'UCR-score assumes that there is exactly a single anomalous event, "
                f"but {len(starts)} anomalous events are given."
            )

        # Retrieve the relevant time points
        t_s = starts[0]
        t_e = ends[0]
        t_star = np.argmax(y_pred)

        # Compute a margin around the anomalous event, based on the tolerance
        margin = max(t_e - t_s + 1, self.tolerance or 0)

        # Check if the anomaly is detected
        if t_s - margin <= t_star <= t_e + margin:
            return 1
        else:
            return 0
