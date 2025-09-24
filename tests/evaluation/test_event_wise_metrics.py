import numpy as np
import pytest

from dtaianomaly.evaluation import EventWiseFBeta, EventWisePrecision, EventWiseRecall
from dtaianomaly.evaluation._event_wise_metrics import _compute_event_wise_metrics


class TestComputeEventWiseMetrics:

    def test_perfect_match(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0])
        # GT Events: (1,2), (5,5) -> Total 2
        # Pred Segments: (1,2), (5,5) -> Total 2
        # TPE: (1,2) overlaps (1,2). (5,5) overlaps (5,5). -> TPE=2
        # FNE: 0
        # FPE: 0
        # N_points = 4 (indices 0, 3, 4, 6)
        # FP_points = 0
        # FAR = 0/4 = 0
        # RE = 2/2 = 1.0
        # PE = (2 / (2 + 0)) * (1 - 0) = 1.0 * 1.0 = 1.0
        # F1E = 2 * (1 * 1) / (1 + 1) = 1.0
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 1.0
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_all_zeros(self):
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        # GT Events: 0
        # Pred Segments: 0
        # TPE: 0
        # FNE: 0
        # FPE: 0
        # N_points = 10
        # FP_points = 0
        # FAR = 0/10 = 0
        # RE = 1.0 (0/0 -> treat as 1.0 according to implementation logic)
        # PE = 0.0 (denominator 0 -> treat as 0.0)
        # F1E = 0.0 (2*0*1 / (0+1))
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.0
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_all_ones(self):
        y_true = np.ones(10)
        y_pred = np.ones(10)
        # GT Events: (0, 10) -> Total 1
        # Pred Segments: (0, 10) -> Total 1
        # TPE: 1
        # FNE: 0
        # FPE: 0
        # N_points = 0
        # FP_points = 0
        # FAR = 0 (N_points is 0)
        # RE = 1/1 = 1.0
        # PE = (1 / (1 + 0)) * (1 - 0) = 1.0
        # F1E = 2 * (1 * 1) / (1 + 1) = 1.0
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 1.0
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_no_true_anomalies_fp_pred(self):
        y_true = np.zeros(10)
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        # GT Events: 0
        # Pred Segments: (2, 3), (7, 7) -> Total 2
        # TPE: 0
        # FNE: 0
        # FPE: 2 (since no GT events exist)
        # N_points = 10
        # FP_points = 3 (indices 2, 3, 7)
        # FAR = 3/10 = 0.3
        # RE = 1.0 (0/0 -> treat as 1.0)
        # PE = (0 / (0 + 2)) * (1 - 0.3) = 0 * 0.7 = 0.0
        # F1E = 0.0 (because PE is 0)
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.0
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_all_anomalies_missed(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
        y_pred = np.zeros(10)
        # GT Events: (2, 3), (7, 7) -> Total 2
        # Pred Segments: 0
        # TPE: 0
        # FNE: 2
        # FPE: 0
        # N_points = 7 (indices 0, 1, 4, 5, 6, 8, 9)
        # FP_points = 0
        # FAR = 0/7 = 0
        # RE = 0/2 = 0.0
        # PE = 0.0 (denominator TPE+FPE is 0 -> treat as 0.0)
        # F1E = 0.0 (because RE is 0)
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.0
        assert pytest.approx(recall_event) == 0.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_predict_all_ones_some_true(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 0])
        y_pred = np.ones(7)
        # GT Events: (1,2), (5,5) -> Total 2
        # Pred Segments: (0,6) -> Total 1
        # TPE: (0,6) overlaps (1,2). (0,6) overlaps (5,5). Both GT events detected. -> TPE=2
        # FNE: 0
        # FPE: 0 (The single pred segment (0,6) overlaps with GT events, so it's not FPE)
        # N_points = 4 (indices 0, 3, 4, 6)
        # FP_points = 4 (predicted 1 at indices 0, 3, 4, 6 where true is 0)
        # FAR = 4/4 = 1.0
        # RE = 2/2 = 1.0
        # PE = (2 / (2 + 0)) * (1 - 1.0) = 1.0 * 0.0 = 0.0
        # F1E = 0.0 (because PE is 0)
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.0
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_partial_overlap_fp_fn(self):
        y_true = np.array(
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
        )  # GT: (2,4), (8,9) -> Total 2
        y_pred = np.array(
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        )  # Pred: (1,3), (6,8), (11,11) -> Total 3
        # Overlaps: (1,3) overlaps (2,4). (6,8) overlaps (8,9).
        # TPE = 2
        # FNE = 0
        # FPE: (11, 11) does not overlap any GT event. -> FPE=1
        # N_points = 7 (indices 0, 1, 5, 6, 7, 10, 11)
        # FP_points: Pred=1, True=0 at indices 1, 6, 7, 11 -> FP_points=4
        # FAR = 4 / 7
        # RE = 2 / 2 = 1.0
        # PE = (2 / (2 + 1)) * (1 - 4/7) = (2/3) * (3/7) = 2/7
        # F1E = 2 * (1.0 * 2/7) / (1.0 + 2/7) = (4/7) / (9/7) = 4/9
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 2 / 7
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_one_gt_multiple_pred_overlap(self):
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])  # GT: (2, 6) -> Total 1
        y_pred = np.array(
            [0, 0, 1, 1, 0, 1, 1, 0, 0]
        )  # Pred: (2, 3), (5, 6) -> Total 2
        # Overlaps: Both (2,3) and (5,6) overlap with (2,6).
        # TPE = 1 (The single GT event is detected)
        # FNE = 0
        # FPE = 0 (Both pred segments overlap the GT event)
        # N_points = 4 (indices 0, 1, 7, 8)
        # FP_points = 0
        # FAR = 0 / 4 = 0
        # RE = 1 / 1 = 1.0
        # PE = (1 / (1 + 0)) * (1 - 0) = 1.0
        # F1E = 1.0
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 1.0
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_multiple_gt_one_pred_overlap_all(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # GT: (1, 2), (5, 6) -> Total 2
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 0])  # Pred: (1, 6) -> Total 1
        # Overlaps: (1,6) overlaps both (1,2) and (5,6).
        # TPE = 2
        # FNE = 0
        # FPE = 0 (The single pred segment overlaps GT events)
        # N_points = 3 (indices 0, 3, 4, 7) - mistake here, indices 0, 3, 4, 7 => 4 points
        # FP_points: Pred=1, True=0 at indices 3, 4 -> FP_points=2
        # FAR = 2 / 4 = 0.5
        # RE = 2 / 2 = 1.0
        # PE = (2 / (2 + 0)) * (1 - 0.5) = 1.0 * 0.5 = 0.5
        # F1E = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 2/3
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.5
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_multiple_gt_one_pred_overlap_partial(self):
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # GT: (1, 2), (5, 6) -> Total 2
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0])  # Pred: (2, 4) -> Total 1
        # Overlaps: (2,4) overlaps (1,2). It does NOT overlap (5,6).
        # TPE = 1 (Only GT event (1,2) is detected)
        # FNE = 1 (GT event (5,6) is missed)
        # FPE = 0 (The single pred segment (2,4) overlaps with a GT event)
        # N_points = 4 (indices 0, 3, 4, 7)
        # FP_points: Pred=1, True=0 at indices 3, 4 -> FP_points=2
        # FAR = 2 / 4 = 0.5
        # RE = 1 / 2 = 0.5
        # PE = (1 / (1 + 0)) * (1 - 0.5) = 1.0 * 0.5 = 0.5
        # F1E = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 2 * 0.25 / 1.0 = 0.5
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.5
        assert pytest.approx(recall_event) == 0.5
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_pred_segment_between_gt(self):
        y_true = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0])  # GT: (1,2), (6,7) -> Total 2
        y_pred = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0])  # Pred: (1,2), (4,5) -> Total 2
        # Overlaps: (1,2) overlaps (1,2). (4,5) does not overlap any GT event.
        # TPE = 1
        # FNE = 1
        # FPE = 1 (Segment (4,5) is FP)
        # N_points = 5 (indices 0, 3, 4, 5, 8)
        # FP_points: Pred=1, True=0 at indices 4, 5 -> FP_points=2
        # FAR = 2 / 5 = 0.4
        # RE = 1 / 2 = 0.5
        # PE = (1 / (1 + 1)) * (1 - 0.4) = 0.5 * 0.6 = 0.3
        # F1E = 2 * (0.5 * 0.3) / (0.5 + 0.3) = 0.3 / 0.8 = 3/8 = 0.375
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.3
        assert pytest.approx(recall_event) == 0.5
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_pred_touching_edge(self):
        y_true = np.array([0, 1, 1, 1, 0])  # GT: (1, 3) -> Total 1
        y_pred = np.array([1, 1, 0, 0, 0])  # Pred: (0, 1) -> Total 1
        # Overlaps: (0,1) overlaps (1,3) because index 1 is common.
        # TPE = 1
        # FNE = 0
        # FPE = 0
        # N_points = 2 (indices 0, 4)
        # FP_points: Pred=1, True=0 at index 0 -> FP_points=1
        # FAR = 1 / 2 = 0.5
        # RE = 1 / 1 = 1.0
        # PE = (1 / (1 + 0)) * (1 - 0.5) = 1.0 * 0.5 = 0.5
        # F1E = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 2/3
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.5
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_identical_overlap_different_far_1(self):
        # Scenario 1: Low FAR
        y_true = np.array([0, 0, 1, 1, 0, 0])  # GT: (2, 3) -> Total 1
        y_pred = np.array([0, 1, 1, 1, 1, 0])  # Pred: (1, 4) -> Total 1
        # TPE=1, FNE=0, FPE=0
        # N=4 (0,1,4,5), FP=2 (1,4), FAR=0.5
        # RE=1.0, PE=(1/1)*(1-0.5)=0.5, F1E=2/3
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 0.5
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event

    def test_identical_overlap_different_far_2(self):
        # Scenario 2: High FAR (same event overlap, more normal points hit)
        y_true = np.array([0, 0, 1, 1, 0, 0, 0, 0])  # GT: (2, 3) -> Total 1
        y_pred = np.array([0, 1, 1, 1, 1, 0, 1, 1])  # Pred: (1, 4), (6, 7) -> Total 2
        # TPE=1 (from (1,4) overlapping (2,3))
        # FNE=0
        # FPE=1 (segment (6,7) is FP)
        # N=6 (0,1,4,5,6,7), FP=4 (1,4,6,7), FAR=4/6 = 2/3
        # RE=1.0
        # PE = (1 / (1 + 1)) * (1 - 2/3) = 0.5 * (1/3) = 1/6
        # F1E = 2 * (1.0 * 1/6) / (1.0 + 1/6) = (1/3) / (7/6) = (1/3) * (6/7) = 2/7
        precision_event, recall_event = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        assert pytest.approx(precision_event) == 1 / 6
        assert pytest.approx(recall_event) == 1.0
        assert EventWisePrecision().compute(y_true, y_pred) == precision_event
        assert EventWiseRecall().compute(y_true, y_pred) == recall_event


class TestMetrics:
    # From TestComputeEventWiseMetrics.test_pred_segment_between_gt
    y_true = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0])

    def test_precision(self):
        assert EventWisePrecision().compute(self.y_true, self.y_pred) == 0.3

    def test_recall(self):
        assert EventWiseRecall().compute(self.y_true, self.y_pred) == 0.5

    def test_f05(self):
        # ((1+0.5²) * 0.3 * 0.5) / (0.5² * 0.3 + 0.5)
        # = (1.25 * 0.15) / (0.25 * 0.3 + 0.5) = 0.1875 / 0.575 = 1875/5750
        assert EventWiseFBeta(beta=0.5).compute(
            self.y_true, self.y_pred
        ) == pytest.approx(1875 / 5750)

    def test_f1(self):
        # ((1+1²) * 0.3 * 0.5) / (1² * 0.3 + 0.5)
        # = (2 * 0.15) / 0.8 = 0.3 / 0.8 = 3/8
        assert EventWiseFBeta().compute(self.y_true, self.y_pred) == pytest.approx(
            3 / 8
        )

    def test_f2(self):
        # ((1+2²) * 0.3 * 0.5) / (2² * 0.3 + 0.5)
        # = (5 * 0.15) / (4 * 0.3 + 0.5) = 0.75 / 1.7 = 75/170
        assert EventWiseFBeta(beta=2).compute(
            self.y_true, self.y_pred
        ) == pytest.approx(75 / 170)
