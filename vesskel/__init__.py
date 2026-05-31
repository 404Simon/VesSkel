"""Public package API for vesskel."""

from importlib.metadata import version

from . import thin

__version__ = version("vesskel")
__all__ = ["thin"]
