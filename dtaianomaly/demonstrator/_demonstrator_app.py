import streamlit as st

from dtaianomaly.anomaly_detection import IsolationForest
from dtaianomaly.data import DemonstrationTimeSeriesLoader
from dtaianomaly.demonstrator._footer import footer
from dtaianomaly.demonstrator._header import header
from dtaianomaly.demonstrator._st_AnomalyDetector import StAnomalyDetectorLoader
from dtaianomaly.demonstrator._st_DataLoader import StDataLoader
from dtaianomaly.demonstrator._st_QualitativeEvaluator import StQualitativeEvaluator
from dtaianomaly.utils import all_classes

###################################################################
# LAYOUT
###################################################################

st.set_page_config(
    page_title="dtaianomaly demonstrator",
    page_icon="https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/main/docs/logo/favicon.svg",
    layout="wide",
)
st.logo(
    "https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/main/docs/logo/readme.svg",
    link="https://github.com/ML-KULeuven/dtaianomaly",
    icon_image="https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/main/docs/logo/favicon.svg",
)
header()
footer()


###################################################################
# SESSION STATE CONFIGURATION
###################################################################


if "st_data_loader" not in st.session_state:
    st.session_state.st_data_loader = StDataLoader(
        all_data_loaders=all_classes("data-loader", return_names=True),
        default_data_loader=DemonstrationTimeSeriesLoader(),
    )
if "st_anomaly_detector_loader" not in st.session_state:
    st.session_state.st_anomaly_detector_loader = StAnomalyDetectorLoader(
        all_anomaly_detectors=all_classes("anomaly-detector", return_names=True),
        default_anomaly_detector=IsolationForest,
    )
if "loaded_detectors" not in st.session_state:
    st.session_state.loaded_detectors = [
        st.session_state.st_anomaly_detector_loader.select_default_anomaly_detector()
    ]
    st.session_state.loaded_detectors[0].fit_predict(
        st.session_state.st_data_loader.data_set
    )


###################################################################
# UTILITIES
###################################################################


def write_code_lines(lines):
    if len(lines) == 0:
        return
    with st.expander("Show code for ``dtaianomaly``", icon="💻"):
        st.code(body="\n".join(lines), language="python", line_numbers=True)


###################################################################
# APPLICATION
###################################################################

# Introduction
st.title("Welcome to the ``dtaianomaly`` Demonstrator!")
st.warning("**TODO** Write a short, general introduction", icon="✒️")
st.warning(
    "**TODO** Maybe also include some general advertisement (i.e., KU Leuven, DTAI, M-group, our publication, ...?) -> in header/footer?"
)

# Data
st.header("Time series data")
st.warning("**TODO** Write a short introduction about loading data", icon="✒️")
data_updated = st.session_state.st_data_loader.select_data_loader()
st.session_state.st_data_loader.show_data()
write_code_lines(st.session_state.st_data_loader.get_code_lines())

# Retrain the anomaly detectors if the data was updated
if data_updated:
    for detector in st.session_state.loaded_detectors:
        detector.fit_predict(st.session_state.st_data_loader.data_set)

# Anomaly detection
st.header("Anomaly detection")
st.warning("**TODO** Write a short introduction about anomaly detection", icon="✒️")
new_detector = st.session_state.st_anomaly_detector_loader.select_anomaly_detector()

if new_detector is not None:
    st.session_state.loaded_detectors.append(new_detector)
    new_detector.fit_predict(st.session_state.st_data_loader.data_set)

for i, detector in enumerate(st.session_state.loaded_detectors):
    remove_detector = detector.show_anomaly_detector()
    if remove_detector:
        del st.session_state.loaded_detectors[i]
        st.rerun()  # To make sure that the detector is effectively removed

# Numeric analysis
st.header("Numerical analysis of the anomaly detectors")
st.warning(
    "**TODO** Write a short introduction", icon="✒️"
)  # TODO probably similar to anomaly detectors?

# Visual analysis
st.header("Visual analysis of the anomaly scores")
st.markdown(
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
