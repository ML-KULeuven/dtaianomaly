import streamlit as st

from dtaianomaly.anomaly_detection import IsolationForest
from dtaianomaly.data import DemonstrationTimeSeriesLoader
from dtaianomaly.demonstrator._configuration import load_configuration
from dtaianomaly.demonstrator._st_AnomalyDetector import StAnomalyDetectorLoader
from dtaianomaly.demonstrator._st_DataLoader import StDataLoader
from dtaianomaly.demonstrator._st_QualitativeEvaluator import StQualitativeEvaluator
from dtaianomaly.demonstrator._utils import error_no_detectors, write_code_lines
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


###################################################################
# SESSION STATE CONFIGURATION
###################################################################

if "configuration" not in st.session_state:
    st.session_state.configuration = load_configuration()
if "st_data_loader" not in st.session_state:
    st.session_state.st_data_loader = StDataLoader(
        all_data_loaders=all_classes("data-loader", return_names=True),
        configuration=st.session_state.configuration["data-loader"],
    )
if "st_anomaly_detector_loader" not in st.session_state:
    st.session_state.st_anomaly_detector_loader = StAnomalyDetectorLoader(
        all_anomaly_detectors=all_classes("anomaly-detector", return_names=True),
        configuration=st.session_state.configuration["detector"],
    )
if "loaded_detectors" not in st.session_state:
    st.session_state.loaded_detectors = [
        st.session_state.st_anomaly_detector_loader.select_default_anomaly_detector()
    ]
    st.session_state.loaded_detectors[0].fit_predict(
        st.session_state.st_data_loader.data_set
    )


###################################################################
# INTRODUCTION
###################################################################

st.title("Welcome to the ``dtaianomaly`` Demonstrator!")
st.warning("**TODO** Write a short, general introduction", icon="✒️")
st.warning(
    "**TODO** Maybe also include some general advertisement (i.e., KU Leuven, DTAI, M-group, our publication, ...?) -> in header/footer?"
)

###################################################################
# DATA LOADING
###################################################################

st.header("Time series data")
st.warning("**TODO** Write a short introduction about loading data", icon="✒️")
data_updated = st.session_state.st_data_loader.select_data_loader()
write_code_lines(st.session_state.st_data_loader.get_code_lines())
st.session_state.st_data_loader.show_data()

# Retrain the anomaly detectors if the data was updated
if data_updated:
    for detector in st.session_state.loaded_detectors:
        detector.fit_predict(st.session_state.st_data_loader.data_set)

###################################################################
# ANOMALY DETECTION
###################################################################

st.header("Anomaly detection")
st.warning("**TODO** Write a short introduction about anomaly detection", icon="✒️")
st.warning(
    "**TODO** Is this the best possible layout? Maybe we can do something similar as with the data loader?"
)
new_detector = st.session_state.st_anomaly_detector_loader.select_anomaly_detector()

if new_detector is not None:
    st.session_state.loaded_detectors.append(new_detector)
    new_detector.fit_predict(st.session_state.st_data_loader.data_set)

if len(st.session_state.loaded_detectors) == 0:
    error_no_detectors()

for i, detector in enumerate(st.session_state.loaded_detectors):
    updated_detector, remove_detector = detector.show_anomaly_detector()
    write_code_lines(detector.get_code_lines())

    if remove_detector:
        del st.session_state.loaded_detectors[i]
        st.rerun()  # To make sure that the detector is effectively removed

    if updated_detector:
        detector.fit_predict(st.session_state.st_data_loader.data_set)

###################################################################
# VISUAL ANALYSIS
###################################################################

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

###################################################################
# NUMERICAL ANALYSIS
###################################################################

st.header("Numerical analysis of the anomaly detectors")
st.warning("**TODO** Write a short introduction", icon="✒️")
st.warning(
    "Do we want to select the evaluation metrics, or just show all? Selecting would be nice, but also more complex ... "
)
