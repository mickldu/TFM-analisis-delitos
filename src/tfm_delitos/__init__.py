from importlib.metadata import version, PackageNotFoundError
__all__ = ["__version__"]
try:
    __version__ = version("tfm_delitos")
except PackageNotFoundError:
    __version__ = "0.0.0"
