import ast
import importlib
import inspect
import json
import pathlib

import streamlit as st


def error_no_detectors():
    st.error("There are no anomaly detectors selected", icon="🚨")


def error_no_metrics():
    st.error("There are no evaluation metrics selected", icon="🚨")


def write_code_lines(lines, use_expander: bool = True):
    if len(lines) == 0:
        return
    if use_expander:
        with st.expander("Show code for ``dtaianomaly``", icon="💻"):
            st.code(body="\n".join(lines), language="python", line_numbers=True)
    else:
        st.code(body="\n".join(lines), language="python", line_numbers=True)


def get_class_summary(cls) -> str | None:
    doc = cls.__doc__
    if not doc:
        return None
    # Split by blank lines to get the first paragraph
    paragraphs = doc.split("\n\n")
    if len(paragraphs) < 2:
        return None
    return paragraphs[1] if paragraphs else doc


def show_class_summary(cls) -> None:
    summary = get_class_summary(cls)
    if summary is not None:
        st.markdown(summary)


def show_small_header(o) -> None:
    st.markdown(f"##### {o}")


def show_section_description(s) -> None:
    st.markdown(s)


def get_parameters(cls):
    signature = inspect.signature(cls.__init__)
    params = []
    required_params = []

    for name, param in signature.parameters.items():

        # skip 'self' and 'kwargs'
        if name == "self" or name == "kwargs":
            continue

        # Add the parameter
        params.append(name)

        # Check if the parameter is required
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            required_params.append(name)

    return params, required_params


def remove_set_values(o: object, hyperparameters: dict[str, any]) -> dict[str, any]:
    """
    Given a dictionary which maps attributes to values, return a new dictionary which only contains the
    items that have actually a different value from the one set in the object.
    """
    return {
        param: value
        for param, value in hyperparameters.items()
        if getattr(o, param) != value
    }


def update_object(o: object, hyperparameters: dict[str, any]) -> bool:
    """
    Given a dictionary which maps attributes to values, return a new dictionary which only contains the
    items that have actually a different value from the one set in the object.
    """
    updated = False
    for param, value in hyperparameters.items():
        recursive_params = param.split(".")
        inner_object = o
        for p in recursive_params[:-1]:
            inner_object = getattr(inner_object, p)
        if getattr(inner_object, recursive_params[-1]) != value:
            updated = True
            setattr(inner_object, param, value)
    return updated


def input_widget_hyperparameter(widget_type: str, **kwargs) -> any:
    if widget_type == "number_input":
        return st.number_input(**kwargs)
    elif widget_type == "select_slider":
        return st.select_slider(**kwargs)
    elif widget_type == "toggle":
        return st.toggle(**kwargs)
    elif widget_type == "checkbox":
        return st.checkbox(**kwargs)
    elif widget_type == "pills":
        return st.pills(**kwargs)
    elif widget_type == "segmented_control":
        return st.segmented_control(**kwargs)
    elif widget_type == "selectbox":
        return st.selectbox(**kwargs)
    elif widget_type == "slider":
        return st.slider(**kwargs)


def load_custom_models(custom_models_str: str) -> dict[str, list[(str, type)]]:

    def _load_cls(class_path: str) -> (str, type):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return class_name, getattr(module, class_name)

    custom_models = ast.literal_eval(custom_models_str)
    return {
        "data_loaders": [
            _load_cls(data_loader) for data_loader in custom_models["data_loaders"]
        ],
        "anomaly_detectors": [
            _load_cls(anomaly_detector)
            for anomaly_detector in custom_models["anomaly_detectors"]
        ],
        "metrics": [_load_cls(metric) for metric in custom_models["metrics"]],
        "custom_visualizers": [
            _load_cls(visualizer) for visualizer in custom_models["custom_visualizers"]
        ],
    }
