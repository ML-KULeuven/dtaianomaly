import inspect
import json
import os

import toml

from dtaianomaly import anomaly_detection, data, evaluation, preprocessing, thresholding
from dtaianomaly.utils import convert_to_list
from dtaianomaly.workflow import Workflow

__all__ = ["workflow_from_config", "interpret_config"]


def workflow_from_config(path: str, max_size: int = 1000000):
    """
    Construct a Workflow using a configuration at a given path.

    Construct a Workflow instance based on a JSON or TOML file. The file is
    first parsed, and then interpreted to obtain a :py:class:`~dtaianomaly.workflow.Workflow`

    Parameters
    ----------
    path : str
        Path to the config file.
    max_size : int, optional
        Maximal size of the config file in bytes. Defaults to 1 MB.

    Returns
    -------
    Workflow
        The parsed workflow from the given config file.

    Raises
    ------
    TypeError
        If the given path is not a string.
    FileNotFoundError
        If the given path does not correspond to an existing file.
    ValueError
        If the given path does not refer to a json or TOML file.
    """
    if not isinstance(path, str):
        raise TypeError("Path expects a string")
    if not os.path.exists(path):
        raise FileNotFoundError("The given path does not exist!")

    if path.endswith(".json"):
        with open(path, "r") as file:
            # Check file size
            file.seek(0, 2)
            file_size = file.tell()
            if file_size > max_size:
                raise ValueError(f"File size exceeds maximum size of {max_size} bytes")
            file.seek(0)

            # Parse actual JSON
            parsed_config = json.load(file)

    elif path.endswith(".toml"):
        with open(path, "r") as f:
            parsed_config = toml.load(f)

    else:
        raise ValueError("The given path should be a json or toml file!")

    return interpret_config(parsed_config)


def interpret_config(config: dict) -> Workflow:
    """
    Actual parsing/interpretation logic.

    Interprets a given dictionary, and returns the corresponding
    workflow, setup as defined in the configuration file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to parse.

    Returns
    -------
    Workflow
        A Workflow object containing all the components specified in the config.
    """
    # Check the config file
    if not isinstance(config, dict):
        raise TypeError("Input should be a dictionary")

    return Workflow(
        dataloaders=_interpret_config("dataloaders", config, True),
        preprocessors=_interpret_config("preprocessors", config, False),
        detectors=_interpret_config("detectors", config, True),
        metrics=_interpret_config("metrics", config, True),
        thresholds=_interpret_config("thresholds", config, False),
        **_interpret_additional_information(config),
    )


def _interpret_config(name, config, required: bool):
    if name not in config:
        if required:
            raise ValueError(
                f"Required item '{name}' is not given in the config, it only contains {set(config.keys())}"
            )
        else:
            return None

    def _flatten(xs):
        flat = []
        for x in xs:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        return flat

    return _flatten(
        list(map(lambda e: _interpret_entry(e), convert_to_list(config[name])))
    )


def _interpret_entry(entry):
    # Handle difficult case because of POS_VAR parameter
    if "type" in entry and entry["type"] == "ChainedPreprocessor":
        if "base_preprocessors" not in entry:
            raise TypeError(
                f"ChainedPreprocessor.__init__() missing 1 required positional argument: 'ChainedPreprocessor'"
            )
        if len(entry) > 2:
            raise TypeError(
                f"ChainedPreprocessor.__init__() got unexpected keyword arguments: {set(k for k in entry if k not in ['type', 'base_preprocessors'])}"
            )
        return preprocessing.ChainedPreprocessor(
            *map(_interpret_entry, entry["base_preprocessors"])
        )

    # Format the entry
    entry_without_type = {key: value for key, value in entry.items() if key != "type"}
    for key, value in entry_without_type.items():
        if isinstance(value, dict):
            entry_without_type[key] = _interpret_entry(value)
        if key == "base_type":
            entry_without_type["base_type"] = getattr(data, entry["base_type"])

    # Search the module and initialize the object
    modules = [data, preprocessing, anomaly_detection, evaluation, thresholding]
    for module in modules:
        if "type" in entry and hasattr(module, entry["type"]):
            return getattr(module, entry["type"])(**entry_without_type)

    # If everything fails, raise an error
    raise ValueError(f"Invalid entry given to interpret: {entry}")


def _interpret_additional_information(config):
    return {
        argument: config[argument]
        for argument in inspect.signature(Workflow.__init__).parameters.keys()
        if argument in config
        and argument
        not in [
            "self",
            "dataloaders",
            "metrics",
            "detectors",
            "preprocessors",
            "thresholds",
        ]
    }
