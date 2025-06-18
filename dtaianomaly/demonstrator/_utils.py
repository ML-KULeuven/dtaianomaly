import inspect

import streamlit as st


def error_no_detectors():
    st.error("There are no anomaly detectors selected", icon="🚨")


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
