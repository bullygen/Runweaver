"""Runweaver: durable typed pipelines and iterative computational experiments."""

from importlib.metadata import PackageNotFoundError, version

from .api import *  # noqa: F403
from .api import __all__ as __all__

try:
    __version__ = version("runweaver")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"
