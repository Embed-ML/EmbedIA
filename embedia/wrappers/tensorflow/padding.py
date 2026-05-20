"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow padding layer wrapper.

Covers:
- ZeroPadding1D
- ZeroPadding2D
- ZeroPadding3D
"""

from embedia.wrappers.neural_net_base import PaddingWrapperBase
from embedia.wrappers.tensorflow.base import TensorflowWrapper


class TFPaddingWrapper(TensorflowWrapper, PaddingWrapperBase):
    """
    TensorFlow ZeroPadding layer wrapper for EmbedIA.

    Wraps keras.layers.ZeroPaddingND (1D, 2D, 3D) layers.
    """

    @property
    def padding(self):
        """
        Padding specification from the ZeroPadding layer.

        Can be:
        - int: same padding on all sides
        - tuple of ints: padding for each dimension
        - tuple of tuples: (start, end) padding for each dimension
        """
        return self._target.padding

