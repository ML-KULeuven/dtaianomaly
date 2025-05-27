import numba as nb
import numpy as np

from dtaianomaly.evaluation._common import FBetaBase
from dtaianomaly.evaluation.metrics import BinaryMetric


@nb.njit(fastmath=True, cache=True)
def np_diff(x: np.array):
    diff = np.empty(shape=(x.shape[0] + 1))
    diff[1:-1] = x[1:] - x[:-1]
    diff[0] = x[0]
    diff[-1] = -x[-1]
    return diff


@nb.njit(fastmath=True, cache=True)
def np_any_axis0(x):
    """Numba compatible version of np.any(x, axis=0)."""
    out = np.zeros(x.shape[1], dtype=nb.bool)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out


@nb.njit(fastmath=True, cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=nb.bool)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@nb.njit(fastmath=True, cache=True)
def _make_intervals(y: np.array) -> (np.array, np.array):
    y = (y > 0).astype(np.int8)
    change_points = np_diff(y)
    starts = np.where(change_points == 1)[0]
    ends = np.where(change_points == -1)[0] - 1
    return starts, ends


@nb.njit(fastmath=True, cache=True, parallel=True)
def _compute_event_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray):

    # --- 1. Point-Wise Calculations ---
    fp = np.sum((~y_true) & y_pred)  # Using boolean operators for clarity
    tn = np.sum((~y_true) & (~y_pred))

    # --- 2. Identify Segments/Events ---
    gt_starts, gt_ends = _make_intervals(y_true)
    pred_starts, pred_ends = _make_intervals(y_pred)

    num_gt_events = gt_starts.shape[0]
    num_pred_events = pred_starts.shape[0]

    # Handle edge cases early
    if num_gt_events == 0:
        return 0.0, 1.0
    if num_pred_events == 0:
        return 0.0, 0.0

    # Build interval overlap matrix for efficient calculation
    # This avoids repeated overlap calculations
    overlap_matrix = np.zeros(shape=(num_gt_events, num_pred_events), dtype=nb.bool)

    for i in nb.prange(num_gt_events):
        for j in nb.prange(num_pred_events):

            # Calculate overlap
            start_overlap = max(gt_starts[i], pred_starts[j])
            end_overlap = min(gt_ends[i], pred_ends[j])

            # Simple overlap check
            overlap_matrix[i, j] = start_overlap <= end_overlap

    # Count true positives - each GT event detected at least once
    gt_detected = np_any_axis1(overlap_matrix)
    tpe = np.sum(gt_detected)

    # Count false positives - predicted events that don't overlap with any GT
    pred_is_fp = ~np_any_axis0(overlap_matrix)
    fpe = np.sum(pred_is_fp)

    # Calculate metrics
    recall_event = tpe / num_gt_events

    # Point-level False Alarm Rate
    num_actual_negatives = tn + fp
    far_pt = fp / num_actual_negatives if num_actual_negatives > 0 else 0.0

    # Event-wise precision
    precision_event_ratio = tpe / (tpe + fpe) if (tpe + fpe) > 0 else 0.0
    precision_event = precision_event_ratio * (1.0 - far_pt)
    precision_event = max(0.0, precision_event)  # Guard against negative values

    return precision_event, recall_event


class EventWisePrecision(BinaryMetric):
    """
    Computes the Event-Wise Precision score :cite:`el2024multivariate`.

    Precision measures how accurately the model identifies anomalies.
    For the Event-Wise Precision, the true and false positives are
    considered at the event-level:

    - :math:`TP_e`: the number of ground truth anomalous events that fully or partially
      overlap with a detected segment.
    - :math:`FP_e`: the number of detected segments that do not overlap with any ground
      truth anomalous event.

    The precision is corrected by the false-alarm rate (FAR) to avoid a model which predicts
    all observations as anomalous to have a high score. The FAR is computed on the point-level:

    - :math:`FP`: the number of detected anomalous **points** that are not actually anomalous.

    We then compute the Event-Wise Precision as (with :math:`N`: the total number of normal points):

    .. math::

       \\text{Event-Wise Precision} = \\frac{TP_e}{TP_e + FP_e} \\times (1 - \\frac{FP}{N})
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        event_wise_precision, _ = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        return event_wise_precision


class EventWiseRecall(BinaryMetric):
    """
    Computes the Event-Wise Recall score :cite:`el2024multivariate`.

    Recall measures the model's ability to correctly identify all actual
    anomalies. For the Event-Wise Recall, the true positives and false
    negatives are considered at the event-level:

    - :math:`TP_e`: the number of ground truth anomalous events that fully or partially
      overlap with a detected segment.
    - :math:`FN_e`: the number of ground truth anomalous events that do not overlap with
      a detected segment.

    We then compute the Event-Wise Recall as:

    .. math::

       \\text{Event-Wise Recall} = \\frac{TP_e}{TP_e + FN_e}
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        _, event_wise_recall = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        return event_wise_recall


class EventWiseFBeta(BinaryMetric, FBetaBase):
    """
    Computes the Event-Wise :math:`F_\\beta` score :cite:`el2024multivariate`.

    The :math:`F_\\beta` combines both precision and recall into a single
    value. It provides a balanced evaluation of a model’s performance,
    especially in anomaly detection, where there is often a trade-off
    between catching all anomalies (high recall) and minimizing false
    alarms (high precision). The parameter :math:`\\beta` controls the balance
    between precision and recall. A :math:`\\beta > 1` gives more weight to
    recall, useful when missing anomalies is costly, while :math:`\\beta < 1`
    emphasizes precision, reducing false positives.

    The :math:`F_\\beta` score is the harmonic mean of the Event-Wise Precision
    and Event-Wise Recall.

    Parameters
    ----------
    beta: int, float, default=1
        Desired beta parameter.

    See also
    --------
    EventWisePrecision: Compute the Event-Wise Precision score.
    EventWiseRecall: Compute the Event-Wise Recall score.
    """

    def __init__(self, beta: (float, int) = 1) -> None:
        super().__init__(beta)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        event_wise_precision, event_wise_recall = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        return self._f_score(precision=event_wise_precision, recall=event_wise_recall)
