import importlib.metadata as _metadata

try:
    __version__ = _metadata.version("dtaianomaly")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
