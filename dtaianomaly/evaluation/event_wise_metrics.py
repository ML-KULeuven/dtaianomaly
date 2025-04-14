import numpy as np

from dtaianomaly.evaluation.metrics import BinaryMetric


def _make_intervals(y: np.array):
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    if len(y) == 0:
        return []  # Handle empty array case

    y = (y > 0).astype(np.int8)

    change_points = np.diff(y, prepend=0, append=0)
    starts = np.where(change_points == 1)[0]
    ends = np.where(change_points == -1)[0] - 1

    # Return empty list if no intervals found
    if len(starts) == 0:
        return []

    return list(zip(starts, ends))


def _compute_event_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays of the same shape.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0s and 1s")
    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("y_pred must contain only 0s and 1s")

    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    # --- 1. Point-Wise Calculations ---
    fp = np.sum((~y_true) & y_pred)  # Using boolean operators for clarity
    tn = np.sum((~y_true) & (~y_pred))

    # --- 2. Identify Segments/Events ---
    gt_events = _make_intervals(y_true)
    pred_events = _make_intervals(y_pred)

    num_gt_events = len(gt_events)
    num_pred_events = len(pred_events)

    # Handle edge cases early
    if num_gt_events == 0:
        return 0.0, 1.0

    if num_pred_events == 0:
        return 0.0, 0.0

    # Build interval overlap matrix for efficient calculation
    # This avoids repeated overlap calculations
    overlap_matrix = np.zeros((num_gt_events, num_pred_events), dtype=bool)

    for i, (gt_start, gt_end) in enumerate(gt_events):
        for j, (pred_start, pred_end) in enumerate(pred_events):
            # Calculate overlap
            start_overlap = max(gt_start, pred_start)
            end_overlap = min(gt_end, pred_end)
            # Simple overlap check
            overlap_matrix[i, j] = start_overlap <= end_overlap

    # Count true positives - each GT event detected at least once
    gt_detected = np.any(overlap_matrix, axis=1)
    tpe = np.sum(gt_detected)

    # Count false positives - predicted events that don't overlap with any GT
    pred_is_fp = ~np.any(overlap_matrix, axis=0)
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
    Computes the Event-Wise Precision score [el2024multivariate]_.

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

    We then compute the Event-Wise Precision as (with :math:`N`: the total number of points):

    .. math::

       \\text{Event-Wise Precision} = \\frac{TP_e}{TP_e + FP_e} \\times (1 - \\frac{FP}{N})
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        event_wise_precision, _ = _compute_event_wise_metrics(
            y_true.astype(int), y_pred.astype(int)
        )
        return event_wise_precision


class EventWiseRecall(BinaryMetric):
    """
    Computes the Event-Wise Recall score [el2024multivariate]_.

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
            y_true.astype(int), y_pred.astype(int)
        )
        return event_wise_recall


class EventWiseFBeta(BinaryMetric):
    """
    Computes the Event-Wise :math:`F_\\beta` score [el2024multivariate]_.

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

    beta: float

    def __init__(self, beta: float = 1.0) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("`beta` should be numeric")
        if beta <= 0.0:
            raise ValueError("`beta` should be strictly positive")
        self.beta = beta

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        event_wise_precision, event_wise_recall = _compute_event_wise_metrics(
            y_true.astype(int), y_pred.astype(int)
        )

        numerator = (1 + self.beta**2) * event_wise_precision * event_wise_recall
        denominator = self.beta**2 * event_wise_precision + event_wise_recall
        return 0.0 if denominator == 0 else numerator / denominator
