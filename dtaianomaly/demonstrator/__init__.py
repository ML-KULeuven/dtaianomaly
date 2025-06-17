import pathlib
import sys

import torch
from streamlit.web import cli as stcli

from ._configuration import load_configuration, reset_configuration, set_configuration

torch.classes.__path__ = []  # To avoid torch-warning


__all__ = ["run", "load_configuration", "set_configuration", "reset_configuration"]


def run():
    sys.argv = [
        "streamlit",
        "run",
        pathlib.Path(__file__).parent / "_demonstrator_app.py",
    ]
    sys.exit(stcli.main())
