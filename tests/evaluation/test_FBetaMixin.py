import pytest

from dtaianomaly.evaluation import FBetaMixin


class TestFBetaMixin:

    @pytest.mark.parametrize(
        "beta,precision,recall,expected",
        [
            (1, 1.0, 1.0, 1.0),
            (0.5, 1.0, 1.0, 1.0),
            (2, 1.0, 1.0, 1.0),
            # Zero precision → score always 0 (unless recall is also 0 → defined as 0)
            (1, 0.0, 0.8, 0.0),
            (2, 0.0, 1.0, 0.0),
            # Zero recall → score always 0
            (1, 0.9, 0.0, 0.0),
            (0.5, 0.7, 0.0, 0.0),
            # Balanced case: precision = recall → F = same as that value
            (1, 0.6, 0.6, 0.6),
            (2, 0.4, 0.4, 0.4),
            # Precision higher than recall (β > 1 emphasizes recall)
            (2, 0.9, 0.5, (1 + 4) * 0.9 * 0.5 / (4 * 0.9 + 0.5)),  # ≈ 0.5556
            (0.5, 0.9, 0.5, (1 + 0.25) * 0.9 * 0.5 / (0.25 * 0.9 + 0.5)),  # ≈ 0.8182
            # Recall higher than precision
            (2, 0.5, 0.9, (1 + 4) * 0.5 * 0.9 / (4 * 0.5 + 0.9)),  # ≈ 0.7895
            (0.5, 0.5, 0.9, (1 + 0.25) * 0.5 * 0.9 / (0.25 * 0.5 + 0.9)),  # ≈ 0.5172
            # Edge case: both precision and recall = 0
            (1, 0.0, 0.0, 0.0),
        ],
    )
    def test(self, beta, precision, recall, expected):
        assert FBetaMixin(beta=beta)._f_score(precision, recall) == expected
