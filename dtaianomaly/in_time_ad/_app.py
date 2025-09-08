import os.path
import sys
import warnings

import streamlit as st

from dtaianomaly.in_time_ad._configuration import load_configuration
from dtaianomaly.in_time_ad._st_AnomalyDetector import StAnomalyDetectorLoader
from dtaianomaly.in_time_ad._st_DataLoader import StDataLoader
from dtaianomaly.in_time_ad._st_QualitativeEvaluator import StQualitativeEvaluator
from dtaianomaly.in_time_ad._st_QuantitativeEvaluator import (
    StEvaluationScores,
    StQualitativeEvaluationLoader,
)
from dtaianomaly.in_time_ad._utils import (
    error_no_detectors,
    error_no_metrics,
    load_custom_models,
    show_header,
    show_section_description,
    write_code_lines,
)
from dtaianomaly.utils import all_classes

###################################################################
# LAYOUT
###################################################################

st.set_page_config(
    page_title="InTimeAD",
    page_icon="https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/main/docs/logo/favicon.svg",
    layout="wide",
)
st.logo(
    "https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/main/docs/logo/readme.svg",
    link="https://github.com/ML-KULeuven/dtaianomaly",
    icon_image="https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/main/docs/logo/favicon.svg",
)


###################################################################
# SESSION STATE CONFIGURATION
###################################################################

if "configuration" not in st.session_state:
    config_path = sys.argv[1]
    if config_path == "default":
        config = load_configuration()
    elif not os.path.isfile(config_path):
        warnings.warn(
            f"The given configuration file does not exist: '{config_path}'. Using the default configuration."
        )
    else:
        config = load_configuration(config_path)
    st.session_state.configuration = load_configuration()
if "custom_models" not in st.session_state:
    st.session_state.custom_models = load_custom_models(sys.argv[2])
if "st_data_loader" not in st.session_state:
    st.session_state.st_data_loader = StDataLoader(
        all_data_loaders=all_classes(type_filter="data-loader", return_names=True)
        + st.session_state.custom_models["data_loaders"],
        configuration=st.session_state.configuration["data-loader"],
    )
if "st_anomaly_detector_loader" not in st.session_state:
    st.session_state.st_anomaly_detector_loader = StAnomalyDetectorLoader(
        all_anomaly_detectors=all_classes(
            type_filter="anomaly-detector", return_names=True
        )
        + st.session_state.custom_models["anomaly_detectors"],
        all_custom_detector_visualizers=all_classes(
            type_filter="custom-demonstrator-visualizers", return_names=True
        )
        + st.session_state.custom_models["custom_visualizers"],
        configuration=st.session_state.configuration["detector"],
    )
if "loaded_detectors" not in st.session_state:
    st.session_state.loaded_detectors = (
        st.session_state.st_anomaly_detector_loader.select_default_anomaly_detector()
    )
    for st_detector in st.session_state.loaded_detectors:
        st_detector.fit_predict(st.session_state.st_data_loader.data_set)
if "st_metric_loader" not in st.session_state:
    st.session_state.st_metric_loader = StQualitativeEvaluationLoader(
        all_metrics=all_classes(type_filter="metric", return_names=True)
        + st.session_state.custom_models["metrics"],
        configuration=st.session_state.configuration["metric"],
    )
if "loaded_metrics" not in st.session_state:
    st.session_state.loaded_metrics = (
        st.session_state.st_metric_loader.select_default_metrics()
    )
if "st_evaluation_scores" not in st.session_state:
    st.session_state.st_evaluation_scores = StEvaluationScores(
        detectors=st.session_state.loaded_detectors,
        metrics=st.session_state.loaded_metrics,
        y_test=st.session_state.st_data_loader.data_set.y_test,
    )


###################################################################
# INTRODUCTION
###################################################################

st.title("Welcome to ``InTimeAD``!")
st.subheader("Interactive Time Series Anomaly Detection")
show_section_description(
    """
    InTimeAD is tool for *In*teractive *Time* Series *A*nomaly *D*etection, which offers a
    simple webinterface to apply state-of-the-art time series anomaly detection. InTimeAD
    builds on [``dtaianomaly``](https://github.com/ML-KULeuven/dtaianomaly), an easy-to-use
    Python package for time series anomaly detection, to let you detect anomalies without
    writing any code. This makes it possible to quickly explore and compare different
    models, including on your own data. Once you’ve identified suitable models, you can
    switch to Python for more in-depth validation. To help with this transition, code
    snippets are provided throughout this InTimeAD (marked with a "💻") and can be copy-pasted
    directly in your code-base.
    """
)
with st.expander("What is anomaly detection?", expanded=False, icon="💡"):
    show_section_description(
        """
        A **time series** is an ordered sequence of observations measured over time.
        For example, the CPU usage of a server recorded every minute. A time series
        is **univariate** if it tracks only a single variable (e.g., just CPU usage), or
        **multivariate** if it tracks 2 or more variables simultaneously (e.g., CPU usage,
        memory load, network traffic, ...).

        An **anomaly** in a time series is an observation or a sequence of observations that
        deviate from the normal behavior, from the expected. For example, high CPU usage in
        off-peak hours may indicate a security breach. Anomalies can signal system malfunctioning
        or other critical issues that need to be resolved.

        **Time series anomaly detection** is the task of automatically identifying these unexpected
        patterns. The automated detection of anomalies helps to maintain system health, to reduce
        downtime, and to improve reliability.

        Typically, an anomaly detection model will compute continuous **anomaly scores**: a
        numeric value for each observation in the time series which indicates how anomalous that
        observation is. For anomalous measurements, the anomaly score will be large, while the
        score will be small for normal observations.
        """
    )


###################################################################
# DATA LOADING
###################################################################

show_header("Time series data")
show_section_description(
    """
    To get started, load a time series into InTimeAD. You can either use one of
    the built-in data loaders or upload your own data. The time series will be shown
    immediately to help you understand its structure. Once you're familiar with the data,
    you can begin detecting anomalies.
    """
)
data_updated = st.session_state.st_data_loader.select_data_loader()
st.session_state.st_data_loader.show_data()
write_code_lines(st.session_state.st_data_loader.get_code_lines())

# Retrain the anomaly detectors if the data was updated
if data_updated:
    for detector in st.session_state.loaded_detectors:
        detector.fit_predict(st.session_state.st_data_loader.data_set)
        for metric in st.session_state.loaded_metrics:
            st.session_state.st_evaluation_scores.add(
                detector, metric, st.session_state.st_data_loader.data_set.y_test
            )

###################################################################
# ANOMALY DETECTION
###################################################################

show_header("Anomaly detection")
show_section_description(
    """
    Over the years, many anomaly detection models have been developed. Each of these
    models detects anomalies in a different manner, based on different assumptions
    of what constitutes to an anomaly. Below you can select and configure one or
    more anomaly detectors to apply on the time series. All the hyperparameters are
    filled in by default, but you can tune these in order to better detect the
    anomalies or to analyze their effect on the performance.
    """
)

new_detector = st.session_state.st_anomaly_detector_loader.select_anomaly_detector()

if new_detector is not None:
    st.session_state.loaded_detectors.append(new_detector)
    new_detector.fit_predict(st.session_state.st_data_loader.data_set)
    for metric in st.session_state.loaded_metrics:
        st.session_state.st_evaluation_scores.add(
            new_detector, metric, st.session_state.st_data_loader.data_set.y_test
        )

if len(st.session_state.loaded_detectors) == 0:
    error_no_detectors()

for i, detector in enumerate(st.session_state.loaded_detectors):
    updated_detector, remove_detector, old_detector, error_container = (
        detector.show_anomaly_detector()
    )

    if remove_detector:
        st.session_state.st_evaluation_scores.remove_detector(detector)
        del st.session_state.loaded_detectors[i]
        st.rerun()  # To make sure that the detector is effectively removed

    if updated_detector:
        st.session_state.st_evaluation_scores.remove_detector(old_detector)
        detector.fit_predict(st.session_state.st_data_loader.data_set)
        for metric in st.session_state.loaded_metrics:
            st.session_state.st_evaluation_scores.add(
                detector, metric, st.session_state.st_data_loader.data_set.y_test
            )

    if detector.decision_function_ is None:
        error_message = "Something went wrong while detecting anomalies! No predictions are available."
        if hasattr(detector, "exception_"):
            error_message += f"\n\nError message: {detector.exception_}"

        # Handle errors
        error_container.error(error_message, icon="🚨")
    else:
        detector.show_custom_visualizations()
    write_code_lines(detector.get_code_lines(st.session_state.st_data_loader.data_set))


###################################################################
# VISUAL ANALYSIS
###################################################################

show_header("Visual analysis of the anomaly scores")
show_section_description(
    """
    The advantage of time series is that they are inherently visual. Because of this,
    we can easily verify models by simply plotting the data and the predicted
    anomalies. Below, you can analyze the predicted anomaly scores (_"How anomalous is an
    observation?"_) as well as the detected anomalies (_"Is the observation an anomaly or not?_).
    """
)
tab_anomaly_scores, tab_predicted_anomalies = st.tabs(
    ["Anomaly scores", "Detected anomalies"]
)
with tab_anomaly_scores:
    write_code_lines(
        StQualitativeEvaluator.plot_anomaly_scores(
            data_set=st.session_state.st_data_loader.data_set,
            st_anomaly_detectors=st.session_state.loaded_detectors,
        )
    )
with tab_predicted_anomalies:
    write_code_lines(
        StQualitativeEvaluator.plot_detected_anomalies(
            data_set=st.session_state.st_data_loader.data_set,
            st_anomaly_detectors=st.session_state.loaded_detectors,
        )
    )

###################################################################
# NUMERICAL ANALYSIS
###################################################################

show_header("Numerical analysis of the anomaly detectors")
show_section_description(
    """
    While visual inspection gives a good idea of how well a model performs, it's often
    useful to summarize performance with a single score, a task made easy by ``dtaianomaly``.
    Below, you can choose various evaluation metrics, configure them to fit your application,
    and quantitatively assess each anomaly detector. At the bottom, you'll find both the raw
    scores of each metric and model, but also a bar plot for a quick comparison of model performance.
    """
)
new_metric = st.session_state.st_metric_loader.select_metric()

# Add a new metric
if new_metric is not None:
    st.session_state.loaded_metrics.append(new_metric)
    for detector in st.session_state.loaded_detectors:
        st.session_state.st_evaluation_scores.add(
            detector, new_metric, st.session_state.st_data_loader.data_set.y_test
        )

# Cope with issues
if len(st.session_state.loaded_detectors) == 0:
    error_no_detectors()
if len(st.session_state.loaded_metrics) == 0:
    error_no_metrics()

# Show all the metrics
for i, metric in enumerate(st.session_state.loaded_metrics):
    update_metric, remove_metric, old_name_metric = metric.show_metric()
    write_code_lines(metric.get_code_lines(st.session_state.st_data_loader.data_set))

    if remove_metric:
        st.session_state.st_evaluation_scores.remove_metric(metric)
        del st.session_state.loaded_metrics[i]
        st.rerun()  # To make sure that the metric is effectively removed

    if update_metric:
        st.session_state.st_evaluation_scores.remove_metric(old_name_metric)
        for detector in st.session_state.loaded_detectors:
            st.session_state.st_evaluation_scores.add(
                detector, metric, st.session_state.st_data_loader.data_set.y_test
            )

# Show the scores
st.session_state.st_evaluation_scores.show_scores()

###################################################################
# NUMERICAL ANALYSIS
###################################################################

show_header("Acknowledgements")
st.write(
    "If you find ``dtaianomaly`` or ``InTimeAD`` useful for your work, we would appreciate the following [citation](https://arxiv.org/abs/2502.14381):"
)
st.code(
    """
    @article{carpentier2025dtaianomaly,
          title={{dtaianomaly: A Python library for time series anomaly detection}},
          author={Louis Carpentier and Nick Seeuws and Wannes Meert and Mathias Verbeke},
          year={2025},
          eprint={2502.14381},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2502.14381},
    }
    """,
    language="bibtex",
)
st.markdown(
    "> Carpentier, L., Seeuws, N., Meert, W., Verbeke, M.: dtaianomaly: A Python library for time series anomaly detection (2025), https://arxiv.org/abs/2502.14381"
)

cols = st.columns(5, vertical_alignment="bottom")
cols[1].image("https://upload.wikimedia.org/wikipedia/commons/4/49/KU_Leuven_logo.svg")
cols[3].image(
    "https://raw.githubusercontent.com/FHannes/dtai-logo/master/DTAI_Logo.svg"
)
