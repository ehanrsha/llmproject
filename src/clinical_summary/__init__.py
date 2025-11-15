"""Clinical summarization training helpers."""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - best effort metadata lookup
    __version__ = version("clinical-summary")
except PackageNotFoundError:  # pragma: no cover - fallback when not installed as package
    __version__ = "0.0.0"

__all__ = ["__version__"]
