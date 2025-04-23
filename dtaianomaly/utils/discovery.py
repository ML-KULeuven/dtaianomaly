import abc
import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Union

_MODULES_TO_IGNORE = ("pipeline", "utils", "visualisation", "workflow")


def all_classes(
    type_filter: Union[str, type, List[str], List[type]] = None,
    exclude_types: Union[str, type, List[str], List[type]] = None,
    return_names: bool = True,
) -> Union[List[type], List[Tuple[str, type]]]:
    """
    Discover all the classes from ``dtaianomaly`` by crawling the package. This
    method will only return the ``BaseDetector``, ``LazyDataLoader``, ``Metric``,
    ``Preprocessor`` and ``Thresholding`` objects.

    Parameters
    ----------
    type_filter: str, type, list of str or list of type, default=None
        The names or types of the classes that should be returned. If ``None``,
        then no types will be filtered.
    exclude_types: str, type, list of str or list of type, default=None
        The names or types of the classes that should be excluded from
         being returned. If ``None``, then no types will be excluded.
    return_names: bool, default=True
        Whether to return the names and the type of each class (as tuple),
        or only return the type of each class.

    Returns
    -------
    classes: list of tuples (str, type) or list of type
        The discovered classes in dtaianomaly.
    """

    classes = []
    root = str(Path(__file__).parent.parent)

    for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="dtaianomaly."):
        module_parts = module_name.split(".")
        if (
            any(part in _MODULES_TO_IGNORE for part in module_parts)
            or "._" in module_name
        ):
            continue
        module = import_module(module_name)
        clss = inspect.getmembers(module, inspect.isclass)
        clss = [
            (name, est_cls)
            for name, est_cls in clss
            if not name.startswith("_")
            and str(est_cls).startswith("<class 'dtaianomaly")
        ]

        classes.extend(clss)

    classes = set(classes)
    classes = [
        cls for cls in classes if not _is_abstract(cls[1]) and _has_valid_type(cls[1])
    ]

    if type_filter is not None:
        classes = _filter_types(type_filter, classes)
    if exclude_types is not None:
        classes = set(classes) - set(_filter_types(exclude_types, classes))

    classes = sorted(set(classes), key=itemgetter(0))
    if return_names:
        return classes
    else:
        return [cls for _, cls in classes]


def _is_abstract(cls):
    if abc.ABC in cls.__bases__:
        return True
    if not (hasattr(cls, "__abstractmethods__")):
        return False
    if not len(cls.__abstractmethods__):
        return False
    return True


def _has_valid_type(cls: type):
    from dtaianomaly.anomaly_detection import BaseDetector
    from dtaianomaly.data import LazyDataLoader
    from dtaianomaly.evaluation import Metric
    from dtaianomaly.preprocessing import Preprocessor
    from dtaianomaly.thresholding import Thresholding

    return any(
        issubclass(cls, v)
        for v in [BaseDetector, LazyDataLoader, Metric, Preprocessor, Thresholding]
    )


def _filter_types(
    types: Union[str, type, List[str], List[type]], classes: List[Tuple[str, type]]
):
    from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector
    from dtaianomaly.data.LazyDataLoader import LazyDataLoader
    from dtaianomaly.evaluation.metrics import BinaryMetric, Metric, ProbaMetric
    from dtaianomaly.preprocessing.Preprocessor import Preprocessor
    from dtaianomaly.thresholding.thresholding import Thresholding

    _TYPE_FILTERS = {
        "anomaly-detector": BaseDetector,
        "data-loader": LazyDataLoader,
        "metric": Metric,
        "binary-metric": BinaryMetric,
        "proba-metric": ProbaMetric,
        "preprocessor": Preprocessor,
        "thresholder": Thresholding,
    }

    if not isinstance(types, list):
        types = [types]
    else:
        types = list(types)

    filtered_estimators = []
    for t in types:

        if isinstance(t, str):
            if t not in _TYPE_FILTERS:
                raise ValueError(
                    f"Invalid string type given: '{t}'. Valid values are: {list(_TYPE_FILTERS.keys())}"
                )
            filtered_estimators.extend(
                [est for est in classes if issubclass(est[1], _TYPE_FILTERS[t])]
            )

        elif isinstance(t, type):
            filtered_estimators.extend(
                [est for est in classes if issubclass(est[1], t)]
            )

    return filtered_estimators
