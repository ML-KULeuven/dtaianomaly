import pathlib
import sys

from streamlit.web import cli as stcli


def run():
    sys.argv = [
        "streamlit",
        "run",
        pathlib.Path(__file__).parent / "_demonstrator_app.py",
    ]
    sys.exit(stcli.main())
