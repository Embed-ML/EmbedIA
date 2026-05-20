"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow Batch Normalization layer wrapper.
"""

from embedia.wrappers.neural_net_base import BatchNormWrapperBase
from embedia.wrappers.tensorflow.base import TensorflowWrapper


class TFBatchNormWrapper(TensorflowWrapper, BatchNormWrapperBase):
    """
    TensorFlow BatchNormalization layer wrapper for EmbedIA.

    Wraps keras.layers.BatchNormalization and provides access to:
    - Learnable parameters (gamma, beta)
    - Non-trainable statistics (moving_mean, moving_variance)
    - Epsilon for numerical stability

    Batch norm formula:
        output = gamma * (input - moving_mean) / sqrt(moving_variance + epsilon) + beta
    """

    @property
    def gamma(self):
        """
        Scale parameter (learnable).

        Shape: (channels,), dtype float32.
        Retrieved from layer.get_weights()[0].
        """
        return self._target.get_weights()[0]

    @property
    def beta(self):
        """
        Shift parameter (learnable).

        Shape: (channels,), dtype float32.
        Retrieved from layer.get_weights()[1].
        """
        return self._target.get_weights()[1]

    @property
    def moving_mean(self):
        """
        Running mean (non-trainable, updated during training).

        Shape: (channels,), dtype float32.
        Retrieved from layer.get_weights()[2].
        """
        return self._target.get_weights()[2]

    @property
    def moving_variance(self):
        """
        Running variance (non-trainable, updated during training).

        Shape: (channels,), dtype float32.
        Retrieved from layer.get_weights()[3].
        """
        return self._target.get_weights()[3]

    @property
    def epsilon(self):
        """
        Small constant for numerical stability.

        Typical values: 1e-3, 1e-5.
        Retrieved from layer.epsilon attribute.
        """
        return self._target.epsilon

