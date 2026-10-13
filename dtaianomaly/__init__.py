from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dtaianomaly")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
