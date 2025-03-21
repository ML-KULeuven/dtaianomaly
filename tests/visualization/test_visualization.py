
import pytest
import numpy as np
import matplotlib.pyplot as plt
from dtaianomaly import visualization


@pytest.mark.parametrize('plot_function,additional_args,obligated_y_pred', [
    (visualization.plot_demarcated_anomalies, {}, False),
    (visualization.plot_time_series_colored_by_score, {}, False),
    (visualization.plot_anomaly_scores, {}, True),
    (visualization.plot_time_series_anomalies, {}, True),
    (visualization.plot_with_zoom, {'start_zoom': 50, 'end_zoom': 100}, False)
])
class TestPlottingFunctions:

    def test_univariate(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        plot_function(univariate_time_series, y=y, **additional_args)

    def test_multivariate(self, plot_function, additional_args, obligated_y_pred, multivariate_time_series):
        y = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        plot_function(multivariate_time_series, y=y, **additional_args)

    def test_give_time_steps(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        time_steps = visualization.format_time_steps(None, univariate_time_series.shape[0]) * 2 + 10
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        plot_function(univariate_time_series, y=y, time_steps=time_steps, **additional_args)

    def test_given_feature_names_univariate(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        feature_names = ['dimension 0']
        plot_function(univariate_time_series, y=y, feature_names=feature_names, **additional_args)

    def test_given_feature_names_multivariate(self, plot_function, additional_args, obligated_y_pred, multivariate_time_series):
        y = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        feature_names = [f'dimension {i}' for i in range(multivariate_time_series.shape[1])]
        plot_function(multivariate_time_series, y=y, feature_names=feature_names, **additional_args)

    def test_given_feature_names_different_dimensionality(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        feature_names = ['dimension 0', 'dimension 1']
        with pytest.raises(ValueError):
            plot_function(univariate_time_series, y=y, feature_names=feature_names, **additional_args)


@pytest.mark.parametrize('plot_function,additional_args,obligated_y_pred', [
    (visualization.plot_demarcated_anomalies, {}, False),
    (visualization.plot_time_series_colored_by_score, {}, False),
    (visualization.plot_time_series_anomalies, {}, True),
])
class TestGivenAxis:

    def test_given_axis(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        fig = plt.figure()
        plt.axes()
        [ax] = fig.axes
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        given_fig = plot_function(univariate_time_series, y, ax=ax, **additional_args)
        assert given_fig == fig


@pytest.mark.parametrize('plot_function,additional_args,obligated_y_pred', [
    (visualization.plot_demarcated_anomalies, {}, False),
    (visualization.plot_time_series_anomalies, {}, True),
])
class TestNonBinaryPrediction:

    def test_non_binary_y(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        y = np.random.uniform(size=univariate_time_series.shape[0])
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        with pytest.raises(ValueError):
            plot_function(univariate_time_series, y, **additional_args)

    def test_non_binary_y_pred(self, plot_function, additional_args, obligated_y_pred, univariate_time_series):
        if obligated_y_pred:
            y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
            additional_args['y_pred'] = np.random.uniform(size=univariate_time_series.shape[0])
            with pytest.raises(ValueError):
                plot_function(univariate_time_series, y, **additional_args)


@pytest.mark.parametrize('plot_function,additional_args,obligated_y,obligated_y_pred', [
    (visualization.plot_demarcated_anomalies, {}, True, False),
    (visualization.plot_time_series_colored_by_score, {}, True, False),
    (visualization.plot_time_series_anomalies, {}, True, True),
    (lambda X, ax, time_steps=None: None, {}, False, False),  # Test if it is possible to provide no 'y'
])
class TestPlotWithZoom:

    def test(self, plot_function, additional_args, obligated_y, obligated_y_pred, univariate_time_series):
        if obligated_y:
            additional_args['y'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        visualization.plot_with_zoom(univariate_time_series, start_zoom=100, end_zoom=200, method_to_plot=plot_function, **additional_args)

    def test_given_time_steps(self, plot_function, additional_args, obligated_y, obligated_y_pred, univariate_time_series):
        time_steps = visualization.format_time_steps(None, univariate_time_series.shape[0]) * 2 + 10
        if obligated_y:
            additional_args['y'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        if obligated_y_pred:
            additional_args['y_pred'] = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        visualization.plot_with_zoom(univariate_time_series, start_zoom=100, end_zoom=200, time_steps=time_steps, method_to_plot=plot_function, **additional_args)


class TestPlotConfidence:

    def test_univariate(self, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        y_pred = np.random.uniform(size=univariate_time_series.shape[0])
        confidence = np.random.normal(0, 0.05, size=univariate_time_series.shape[0])
        visualization.plot_anomaly_scores(univariate_time_series, y, y_pred, confidence=confidence)

    def test_multivariate(self, multivariate_time_series):
        y = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        y_pred = np.random.uniform(size=multivariate_time_series.shape[0])
        confidence = np.random.normal(0, 0.05, size=multivariate_time_series.shape[0])
        visualization.plot_anomaly_scores(multivariate_time_series, y, y_pred, confidence=confidence)
