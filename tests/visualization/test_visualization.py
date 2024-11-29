
import numpy as np
import matplotlib.pyplot as plt
from dtaianomaly.visualization import plot_time_series_colored_by_score, plot_time_series_anomalies
import pytest

class TestPlotTimeSeriesColoredByScore:

    def test_univariate(self, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        fig = plot_time_series_colored_by_score(univariate_time_series, y)
        assert len(fig.get_axes()[0].lines) == univariate_time_series.shape[0] - 1

    def test_multivariate(self, multivariate_time_series):
        y = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        fig = plot_time_series_colored_by_score(multivariate_time_series, y)
        assert len(fig.get_axes()[0].lines) == (multivariate_time_series.shape[0] - 1) * multivariate_time_series.shape[1]

    def test_given_axis(self, univariate_time_series):
        fig = plt.figure()
        plt.axes()
        [ax] = fig.axes
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        plot_time_series_colored_by_score(univariate_time_series, y, ax=ax)
        assert len(ax.lines) == univariate_time_series.shape[0] - 1

    class TestPlotTimeSeriesAnomalies:

        def test_univariate(self, univariate_time_series):
            y_true = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            y_pred = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            fig = plot_time_series_anomalies(univariate_time_series, y_true, y_pred)
            assert len(fig.get_axes()[0].collections) == 3

        def test_multivariate(self, multivariate_time_series):
            y_true = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
            y_pred = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
            with pytest.raises(ValueError):
                plot_time_series_anomalies(multivariate_time_series, y_true, y_pred)

        def test_given_axis(self, univariate_time_series):
            fig = plt.figure()
            plt.axes()
            [ax] = fig.axes
            y_true = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            y_pred = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            plot_time_series_anomalies(univariate_time_series, y_true, y_pred, ax=ax)
            assert len(ax.collections) == 3

        def test_correct_scatter_counts(self, univariate_time_series):
            y_true = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            y_pred = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)

            TP = (y_true == 1) & (y_pred == 1)
            FP = (y_true == 0) & (y_pred == 1)
            FN = (y_true == 1) & (y_pred == 0)

            fig = plot_time_series_anomalies(univariate_time_series, y_true, y_pred)
            scatter_dots = fig.get_axes()[0].collections

            assert len(scatter_dots[0].get_offsets()) == TP.sum(), "Mismatch in TP count"
            assert len(scatter_dots[1].get_offsets()) == FP.sum(), "Mismatch in FP count"
            assert len(scatter_dots[2].get_offsets()) == FN.sum(), "Mismatch in FN count"

        def test_non_binary_y_pred(self, univariate_time_series):
            y_true = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            y_pred = np.random.uniform(0, 1, size=univariate_time_series.shape[0])
            with pytest.raises(ValueError, match="The predicted anomaly scores must be binary."):
                plot_time_series_anomalies(univariate_time_series, y_true, y_pred)
