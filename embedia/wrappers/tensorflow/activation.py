"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow activation layer wrapper.

Covers all Keras activation layers:
- ReLU, LeakyReLU, PReLU, ELU
- Sigmoid, Tanh, Softmax, Softplus, Softsign
- Hard Sigmoid, Hard Swish, Swish, Gelu, Linear, etc.
"""

import re
from embedia.wrappers.neural_net_base import ActivationWrapperBase
from embedia.wrappers.tensorflow.base import TensorflowWrapper


class TFActivationWrapper(TensorflowWrapper, ActivationWrapperBase):
    """
    TensorFlow activation layer wrapper for EmbedIA.

    Wraps Keras activation layers (ReLU, LeakyReLU, Sigmoid, etc.)
    and provides standardized function name for code generation.
    """

    @property
    def function_name(self) -> str:
        """
        Activation function name in lowercase, without spaces or underscores.

        Examples:
        - ReLU -> "relu"
        - LeakyReLU -> "leakyrelu"
        - Softmax -> "softmax"
        - softmax_v2 -> "softmax" (removes version suffix)

        Handles both:
        - Keras activation layer objects (ReLU, LeakyReLU, etc.)
        - Dense/Conv layers with activation attribute
        """
        if not hasattr(self._target, 'activation'):
            # target is a Keras layer class like ReLU, LeakyReLU, Softmax
            name = self._target.__class__.__name__
        elif hasattr(self._target.activation, '__name__'):
            # target has activation as a function (e.g., Dense with activation=relu)
            name = self._target.activation.__name__
        else:
            # target has activation as an object (e.g., Dense with activation=ReLU())
            name = self._target.activation.__class__.__name__

        # Convert to lowercase and remove version suffix (e.g., "softmax_v2" -> "softmax")
        return re.sub(r'_[^_]*$', '', name.lower())

    @property
    def leakyrelu_alpha(self) -> float:
        """
        Alpha (negative slope) for LeakyReLU.

        Used by code generator to set the alpha parameter.
        Returns the layer's alpha or negative_slope property.
        """
        if hasattr(self._target, 'activation'):
            relu = self._target.activation
        else:
            relu = self._target

        if hasattr(relu, 'negative_slope'):
            return relu.negative_slope
        return relu.alpha

