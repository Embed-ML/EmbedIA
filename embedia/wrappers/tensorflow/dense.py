"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow Dense layer wrapper.
"""

from embedia.wrappers.neural_net_base import DenseWrapperBase
from embedia.wrappers.tensorflow.base import TensorflowWrapper


class TFDenseWrapper(TensorflowWrapper, DenseWrapperBase):
    """
    TensorFlow Dense (fully connected) layer wrapper for EmbedIA.

    Wraps keras.layers.Dense and provides access to weights and biases
    in the standard EmbedIA format.
    """

    @property
    def weights(self):
        """
        Weight matrix from the Dense layer.

        Format: (input_features, output_features), dtype float32.
        Retrieved from layer.get_weights()[0].
        """
        return self._target.get_weights()[0]

    @property
    def biases(self):
        """
        Bias vector from the Dense layer.

        Format: (output_features,), dtype float32.
        Retrieved from layer.get_weights()[1].
        """
        return self._target.get_weights()[1]

