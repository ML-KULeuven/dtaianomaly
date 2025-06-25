import json
import os
import pathlib


def _current_path() -> str:
    return str(pathlib.Path(__file__).parent)


def set_configuration(configuration: dict) -> None:
    with open(f"{_current_path()}/_configuration.json", "w") as f:
        json.dump(configuration, f, indent=4)


def reset_configuration() -> None:
    with open(f"{_current_path()}/_configuration_default.json", "r") as f:
        set_configuration(json.load(f))


def load_configuration() -> dict:
    with open(f"{_current_path()}/_configuration.json", "r") as f:
        return json.load(f)


def configuration_exists() -> bool:
    return os.path.isfile(f"{_current_path()}/_configuration.json")
