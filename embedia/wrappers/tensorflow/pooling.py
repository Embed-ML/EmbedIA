"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow pooling layer wrappers.

Covers:
- MaxPooling1D, MaxPooling2D, MaxPooling3D
- AveragePooling1D, AveragePooling2D, AveragePooling3D
- GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
- GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
"""

from embedia.wrappers.neural_net_base import PoolingWrapperBase
from embedia.wrappers.tensorflow.base import TensorflowWrapper


class TFPoolingWrapper(TensorflowWrapper, PoolingWrapperBase):
    """
    Base wrapper for all TensorFlow pooling layers.

    Supports MaxPooling, AveragePooling, and Global variants in 1D, 2D, and 3D.
    """

    @property
    def padding(self):
        """Standardized padding type."""
        return self._standardize_padding()

    @property
    def strides(self):
        """Stride values from TensorFlow layer."""
        return getattr(self._target, 'strides', None)

    @property
    def pool_size(self):
        """Pool window size from TensorFlow layer."""
        return getattr(self._target, 'pool_size', None)

    @property
    def dimensions(self):
        """Returns dimensionality: 1 for Pooling1D, 2 for Pooling2D, etc."""
        name = type(self._target).__name__.lower()
        if '1d' in name:
            return 1
        elif '2d' in name:
            return 2
        elif '3d' in name:
            return 3
        return None

    @property
    def function_name(self):
        """
        Pooling function name for code generation.

        Extracts the function type from the layer class name:
        - MaxPooling2D -> "max"
        - AveragePooling2D -> "average"
        - GlobalMaxPooling2D -> "global_max"
        - GlobalAveragePooling2D -> "global_average"
        """
        pool_fn = type(self._target).__name__.lower()
        try:
            if "global" in pool_fn:
                # Examples: "GlobalAveragePooling2D" -> "global_average"
                tipo = pool_fn.split("global")[1]
                return "global_" + tipo.split("pooling")[0]
            elif "pooling" in pool_fn:
                # Examples: "MaxPooling2D" -> "max", "AveragePooling2D" -> "average"
                return pool_fn.split("pooling")[0]
            return pool_fn
        except:
            return pool_fn

    @property
    def is_global(self):
        """Whether this is a GlobalPooling layer."""
        name = type(self._target).__name__.lower()
        return "global" in name and "pooling" in name


# Alias for backward compatibility
TFPoolWrapper = TFPoolingWrapper

