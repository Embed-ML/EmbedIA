"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow convolutional layer wrappers.

Covers:
- Conv1D
- Conv2D
- DepthwiseConv2D
- SeparableConv2D
"""

import numpy as np
from embedia.wrappers.neural_net_base import (
    Conv1DWrapperBase,
    Conv2DWrapperBase,
    SeparableConv2DWrapperBase,
    ConvolutionalWrapperBase
)
from embedia.wrappers.tensorflow.base import TensorflowWrapper
from embedia.utils import diagnostics


class TFConvolutionalWrapper(TensorflowWrapper, ConvolutionalWrapperBase):
    """
    Base wrapper for all TensorFlow convolutional layers.

    Common interface for Conv1D, Conv2D, and Separable variants.
    Provides padding, kernel_size, strides, dilation_rate, filters, etc.
    """

    @property
    def padding(self):
        """Standardized padding type from TensorFlow layer."""
        return self._standardize_padding()

    @property
    def kernel_size(self):
        """Kernel size from TensorFlow layer."""
        return self._target.kernel_size

    @property
    def strides(self):
        """Stride values from TensorFlow layer."""
        return self._target.strides

    @property
    def dilation_rate(self):
        """
        Dilation rate from TensorFlow layer.

        Issues warning if dilation > 1, as MCU export doesn't support it.
        """
        value = getattr(self._target, 'dilation_rate', (1,) * self.dimensions)
        if any(d > 1 for d in value):
            diagnostics.warn(
                f"'{self._target.name}': dilation_rate={value} no soportado "
                f"en EmbedIA MCU — se exportará como (1,) * {self.dimensions}."
            )
        return value

    @property
    def filters(self):
        """Number of filters (output channels)."""
        return self._target.filters

    @property
    def use_bias(self):
        """Whether the layer uses bias."""
        return getattr(self._target, 'use_bias', True)

    @property
    def dimensions(self):
        """Returns dimensionality of the convolution (1, 2, or 3)."""
        name = type(self._target).__name__.lower()
        if '1d' in name:
            return 1
        elif '2d' in name:
            return 2
        elif '3d' in name:
            return 3
        return None


class TFConv1DWrapper(TensorflowWrapper, Conv1DWrapperBase):
    """
    TensorFlow Conv1D layer wrapper for EmbedIA.

    Adapts TensorFlow Conv1D layers to EmbedIA format:
    - Converts weight format from TF to EmbedIA convention
    - Extracts layer parameters (kernel_size, stride, padding, etc.)
    - Provides unified interface for code generation

    Weight format conversion:
        TensorFlow:  (kernel_size, input_channels, filters)
        EmbedIA:     (filters, input_channels, kernel_size)
    """

    def _adapt_weights(self, weights):
        """
        Transpose Conv1D weights from TensorFlow to EmbedIA format.

        Parameters
        ----------
        weights : np.ndarray
            TensorFlow weights in format (kernel_size, input_channels, filters)

        Returns
        -------
        np.ndarray
            EmbedIA weights in format (filters, input_channels, kernel_size)
        """
        # Transpose: (kernel_size, input_channels, filters) -> (filters, input_channels, kernel_size)
        return np.transpose(weights, (2, 1, 0))

    @property
    def padding(self):
        """Standardized padding type."""
        return self._standardize_padding()

    @property
    def kernel_size(self):
        """
        Kernel size as integer.

        TensorFlow returns tuple, we extract the first (and only) value.
        """
        ks = self._target.kernel_size
        return ks[0] if isinstance(ks, tuple) else ks

    @property
    def strides(self):
        """
        Stride as integer.

        TensorFlow returns tuple, we extract the first (and only) value.
        """
        s = self._target.strides
        return s[0] if isinstance(s, tuple) else s

    @property
    def dilation_rate(self):
        """
        Dilation rate as integer.

        TensorFlow returns tuple, we extract the first (and only) value.
        Issues warning if > 1 (not supported in MCU export).
        """
        value = getattr(self._target, 'dilation_rate', (1,))
        if any(d > 1 for d in value):
            diagnostics.warn(
                f"'{self._target.name}': dilation_rate={value} no soportado "
                f"en EmbedIA MCU — se exportará como (1,)."
            )
        value = value[0] if isinstance(value, tuple) else value
        return value

    @property
    def filters(self):
        """Number of filters (output channels)."""
        return self._target.filters

    @property
    def use_bias(self):
        """Whether the layer uses bias."""
        return getattr(self._target, 'use_bias', True)

    @property
    def weights(self):
        """
        Conv1D weights in EmbedIA format.

        Shape: (filters, input_channels, kernel_size)
        """
        return self._adapt_weights(self._target.get_weights()[0])

    @property
    def biases(self):
        """
        Bias vector.

        Shape: (filters,)
        """
        return self._target.get_weights()[1]

    @property
    def input_channels(self):
        """Number of input channels from weight shape."""
        return self.weights.shape[1]  # (filters, input_channels, kernel_size)

    def get_output_length(self, input_length):
        """
        Calculate output sequence length based on input and layer parameters.

        Parameters
        ----------
        input_length : int
            Length of input sequence

        Returns
        -------
        int
            Length of output sequence
        """
        from embedia.core.padding_types import PaddingType

        kernel_size = self.kernel_size
        stride = self.strides
        dilation = self.dilation_rate
        padding = self.padding

        # Effective kernel size with dilation
        effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

        if padding == PaddingType.VALID:
            output_length = (input_length - effective_kernel_size) // stride + 1
        elif padding == PaddingType.SAME:
            output_length = (input_length + stride - 1) // stride
        elif padding == PaddingType.CAUSAL:
            output_length = input_length // stride
        else:
            # Fallback for backward compatibility
            output_length = input_length // stride

        return max(1, output_length)

    @property
    def layer_info(self):
        """Summary information about the layer for debugging."""
        return {
            'type': 'Conv1D',
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self._target.padding,
            'dilation_rate': self.dilation_rate,
            'input_channels': self.input_channels,
            'weights_shape': self.weights.shape,
            'biases_shape': self.biases.shape if self.use_bias else None,
            'activation': self.activation.__name__ if self.activation else None
        }

    def __repr__(self):
        """String representation for debugging."""
        return (f"TFConv1DWrapper(filters={self.filters}, kernel_size={self.kernel_size}, "
                f"strides={self.strides}, padding={self._target.padding})")


class TFConv2DWrapper(TensorflowWrapper, Conv2DWrapperBase):
    """
    TensorFlow Conv2D layer wrapper for EmbedIA.

    Adapts TensorFlow Conv2D and DepthwiseConv2D layers to EmbedIA format.

    Weight format conversion:
        TensorFlow:  (height, width, input_channels, filters)
        EmbedIA:     (filters, input_channels, height, width)
    """

    def _adapt_weights(self, weights):
        """
        Transpose Conv2D weights from TensorFlow to EmbedIA format.

        Parameters
        ----------
        weights : np.ndarray
            TensorFlow weights in format (height, width, input_channels, filters)

        Returns
        -------
        np.ndarray
            EmbedIA weights in format (filters, input_channels, height, width)
        """
        return np.transpose(weights, (3, 2, 0, 1))

    @property
    def padding(self):
        """Standardized padding type."""
        return self._standardize_padding()

    @property
    def kernel_size(self):
        """Kernel size from TensorFlow layer."""
        return self._target.kernel_size

    @property
    def strides(self):
        """Stride values from TensorFlow layer."""
        return self._target.strides

    @property
    def dilation_rate(self):
        """
        Dilation rate from TensorFlow layer.

        Issues warning if > 1 (not supported in MCU export).
        """
        value = getattr(self._target, 'dilation_rate', (1, 1))
        if any(d > 1 for d in value):
            diagnostics.warn(
                f"'{self._target.name}': dilation_rate={value} no soportado "
                f"en EmbedIA MCU — se exportará como (1, 1)."
            )
        return value

    @property
    def filters(self):
        """Number of filters (output channels)."""
        return self._target.filters

    @property
    def use_bias(self):
        """Whether the layer uses bias."""
        return getattr(self._target, 'use_bias', True)

    @property
    def weights(self):
        """
        Conv2D weights in EmbedIA format.

        Shape: (filters, input_channels, kernel_height, kernel_width)
        """
        return self._adapt_weights(self._target.get_weights()[0])

    @property
    def biases(self):
        """Bias vector. Shape: (filters,)"""
        return self._target.get_weights()[1]


class TFSeparableConv2DWrapper(TensorflowWrapper, SeparableConv2DWrapperBase):
    """
    TensorFlow SeparableConv2D layer wrapper for EmbedIA.

    Separable convolution decomposes into:
    1. Depthwise convolution (per-channel)
    2. Pointwise convolution (1x1 conv)

    Weight format conversions:
        Depthwise TensorFlow:   (height, width, input_channels, depth_multiplier)
        Depthwise EmbedIA:      (depth_multiplier, input_channels, height, width)

        Pointwise TensorFlow:   (1, 1, input_channels * depth_multiplier, filters)
        Pointwise EmbedIA:      (filters, input_channels * depth_multiplier, 1, 1)
    """

    def _adapt_depthwise_weights(self, weights):
        """
        Transpose depthwise weights from TensorFlow to EmbedIA format.

        Transformation: (h, w, c, d) -> (d, c, h, w)
        """
        return np.transpose(weights, (3, 2, 0, 1))

    def _adapt_pointwise_weights(self, weights):
        """
        Transpose pointwise weights from TensorFlow to EmbedIA format.

        Transformation: (1, 1, c*d, f) -> (f, c*d, 1, 1)
        """
        return np.transpose(weights, (3, 2, 0, 1))

    @property
    def padding(self):
        """Standardized padding type."""
        return self._standardize_padding()

    @property
    def kernel_size(self):
        """Kernel size from TensorFlow layer."""
        return self._target.kernel_size

    @property
    def strides(self):
        """Stride values from TensorFlow layer."""
        return self._target.strides

    @property
    def dilation_rate(self):
        """
        Dilation rate from TensorFlow layer.

        Issues warning if > 1 (not supported in MCU export).
        """
        value = getattr(self._target, 'dilation_rate', (1, 1))
        if any(d > 1 for d in value):
            diagnostics.warn(
                f"'{self._target.name}': dilation_rate={value} no soportado "
                f"en EmbedIA MCU — se exportará como (1, 1)."
            )
        return value

    @property
    def filters(self):
        """Number of filters (output channels)."""
        return self._target.filters

    @property
    def use_bias(self):
        """Whether the layer uses bias."""
        return getattr(self._target, 'use_bias', True)

    @property
    def depth_weights(self):
        """
        Depthwise kernel in EmbedIA format.

        Shape: (depth_multiplier, input_channels, kernel_height, kernel_width)
        """
        weights = self._target.get_weights()
        depthwise_kernel = weights[0]  # TensorFlow format: (h, w, c, d)
        return self._adapt_depthwise_weights(depthwise_kernel)

    @property
    def depth_biases(self):
        """
        Depthwise bias vector.

        TensorFlow SeparableConv2D has no depthwise bias, so returns zeros.
        Shape: (input_channels * depth_multiplier,)
        """
        weights = self._target.get_weights()
        depthwise_kernel = weights[0]
        in_channels = depthwise_kernel.shape[2]  # input channels
        depth_multiplier = depthwise_kernel.shape[3]  # depth_multiplier

        # EmbedIA expects one bias per channel * depth_multiplier
        return np.zeros(in_channels * depth_multiplier, dtype=np.float32)

    @property
    def point_weights(self):
        """
        Pointwise kernel in EmbedIA format.

        Shape: (filters, input_channels * depth_multiplier, 1, 1)
        """
        weights = self._target.get_weights()
        pointwise_kernel = weights[1]  # TensorFlow format: (1, 1, c*d, f)
        return self._adapt_pointwise_weights(pointwise_kernel)

    @property
    def biases(self):
        """
        Pointwise bias vector.

        Shape: (filters,)
        """
        weights = self._target.get_weights()
        if len(weights) == 3:
            return weights[2]  # bias at index 2
        return np.zeros(self._target.filters, dtype=np.float32)

    @property
    def tf_depthwise_shape(self):
        """Original TensorFlow depthwise kernel shape (for debugging)."""
        weights = self._target.get_weights()
        return weights[0].shape if len(weights) > 0 else None

    @property
    def tf_pointwise_shape(self):
        """Original TensorFlow pointwise kernel shape (for debugging)."""
        weights = self._target.get_weights()
        return weights[1].shape if len(weights) > 1 else None

