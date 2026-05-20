"""
embedia/utils/diagnostics.py

Centralized user-facing messages for the EmbedIA export pipeline.
Provides warn(), error(), info() with consistent formatting.
"""
import warnings

# ANSI color codes
_ORANGE  = "\033[38;5;215m"
_RED     = "\033[38;5;196m"
_CYAN    = "\033[38;5;117m"
_RESET   = "\033[0m"
_BOLD    = "\033[1m"


def _formatted_warn(color, prefix, msg):
    """Internal helper — formats and emits a warning with color."""
    original_format = warnings.formatwarning
    try:
        warnings.formatwarning = lambda m, *a, **k: \
            f"{color}{_BOLD}{prefix}{_RESET}{color} {m}{_RESET}\n"
        warnings.warn(msg, UserWarning, stacklevel=3)
    finally:
        warnings.formatwarning = original_format


def warn(msg: str):
    """
    Emit a user-facing warning — non-fatal issue that may affect results.
    Example: unsupported parameter silently defaulted.
    """
    _formatted_warn(_ORANGE, "⚠️  WARNING:", msg)


def error(msg: str):
    """
    Emit a user-facing error — export can continue but results are likely wrong.
    Example: layer not supported, falling back to approximation.
    """
    _formatted_warn(_RED, "❌ ERROR:", msg)


def info(msg: str):
    """
    Emit an informational message — no action needed from the user.
    Example: optimization applied, memory saved.
    """
    _formatted_warn(_CYAN, "ℹ️  INFO:", msg)