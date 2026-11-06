"""distillo: the few things worth learning from a long recording."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .models import Distillation, KeyPoint, Segment, Transcript
from .pipeline import Distiller, Options

try:
    __version__ = version("distillo")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "Distillation",
    "Distiller",
    "KeyPoint",
    "Options",
    "Segment",
    "Transcript",
    "__version__",
]
