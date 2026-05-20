"""
Legacy shim for backward compatibility.

This module warns that it's deprecated and re-exports symbols from
`embedia.wrappers.tensorflow`. Keep this shim while migrating imports;
once all references are updated it can be removed (there is a backup
at `tensorflow_wrappers.py.deprecated`).
"""
import warnings

warnings.warn(
    "embedia.wrappers.tensorflow_wrappers is deprecated; import from "
    "embedia.wrappers.tensorflow instead",
    DeprecationWarning,
)

from embedia.wrappers.tensorflow import *  # noqa: F401,F403

__all__ = getattr(__import__('embedia.wrappers.tensorflow', fromlist=['__all__']), '__all__', [])
